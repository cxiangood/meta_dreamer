import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

# Install interrupt plot handler early so training will save reward plot on Ctrl+C
try:
  # import for side-effects
  import dreamer.tools.interrupt_plotter  # noqa: F401
except Exception:
  # best-effort import; failure shouldn't block training start
  pass

import elements
import embodied
import numpy as np
import portal
import ruamel.yaml as yaml
from PIL import Image


def main(argv=None):
  from .agent import Agent
  [elements.print(line) for line in Agent.banner]

  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
  config = elements.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))

  if 'JOB_COMPLETION_INDEX' in os.environ:
    config = config.update(replica=int(os.environ['JOB_COMPLETION_INDEX']))
  print('Replica:', config.replica, '/', config.replicas)

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  print('Run script:', config.script)
  if not config.script.endswith(('_env', '_replay')):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer

  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  args = elements.Config(
      **config.run,
      replica=config.replica,
      replicas=config.replicas,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      report_length=config.report_length,
      consec_train=config.consec_train,
      consec_report=config.consec_report,
      replay_context=config.replay_context,
  )

  # 检查ckpt是否存在并输出提示
  ckpt_path = os.path.join(config.logdir, 'ckpt')
  if os.path.exists(ckpt_path):
    print('从上一次ckpt继续训练:', ckpt_path)
  else:
    print('未检测到ckpt，重新开始训练:', ckpt_path)

  if config.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_env, config),
        bind(make_env, config),
        bind(make_stream, config),
        bind(make_logger, config),
        args)

  elif config.script == 'parallel_env':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_env(
        bind(make_env, config), config.replica, args, is_eval)

  elif config.script == 'parallel_envs':
    is_eval = config.replica >= args.envs
    embodied.run.parallel.parallel_envs(
        bind(make_env, config), bind(make_env, config), args)

  elif config.script == 'parallel_replay':
    embodied.run.parallel.parallel_replay(
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'replay_eval', 'eval'),
        bind(make_stream, config),
        args)

  else:
    raise NotImplementedError(config.script)


def make_agent(config):
  from .agent import Agent
  env = make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  if config.random_agent:
    return embodied.RandomAgent(obs_space, act_space)
  cpdir = elements.Path(config.logdir)
  cpdir = cpdir.parent if config.replicas > 1 else cpdir
  return Agent(obs_space, act_space, elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def save_log_images(logs, step, outdir):
  """Persist log image tensors into the job log directory.

  We rely on the configured logdir instead of a hard-coded host path so
  checkpoint, metrics, and image artifacts live together and are easy to
  collect or sync.
  """
  os.makedirs(outdir, exist_ok=True)
  # Keep only the most recent episode's images to save disk space.
  try:
    for fname in os.listdir(outdir):
      if fname.endswith('.png'):
        os.remove(os.path.join(outdir, fname))
  except Exception as e:
    print(f"[Save Log Image] cleanup: {e}")
  for k, v in logs.items():
    # Save images that start with 'log/' or contain 'openloop/' (from agent.report)
    # Note: logger.add may add prefix like 'report/' or 'eval/', so we check for 'openloop/' anywhere
    is_image_key = (k.startswith('log/') or 'openloop/' in k) and isinstance(v, np.ndarray)
    if is_image_key:
      arr = v
      if arr.ndim == 5:  # (B,T,H,W,C)
        arr = arr[0, 0]
      elif arr.ndim == 4:  # (T,H,W,C) or (B,H,W,C) or (T, H, B*W, C) for openloop
        # For openloop images: shape is (T, H, B*W, C) - time sequence grid
        if 'openloop/' in k:
          # arr is (T, H, B*W, C), concatenate all time steps vertically to show full sequence
          # Limit to reasonable size: take every frame or sample frames
          T, H, W, C = arr.shape
          if T > 50:  # If too many frames, sample every few frames
            step_size = T // 50
            arr = arr[::step_size]
            T = len(arr)
          # Concatenate all frames vertically: (T*H, B*W, C)
          arr = arr.reshape((T * H, W, C))
        else:
          arr = arr[0]
      if arr.ndim == 3 and arr.shape[-1] in [1, 3, 4]:
        img = arr[..., :3] if arr.shape[-1] > 3 else arr
        img = img.astype(np.uint8)
        fname = os.path.join(outdir, f'{k.replace("/", "_")}_{step:06d}.png')
        try:
          Image.fromarray(img).save(fname)
        except Exception as e:
          print(f"[Save Log Image] {k}: {e}")


def make_logger(config):
  step = elements.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  outputs = []
  outputs.append(elements.logger.TerminalOutput(config.logger.filter, 'Agent'))
  for output in config.logger.outputs:
    if output == 'jsonl':
      outputs.append(elements.logger.JSONLOutput(logdir, 'metrics.jsonl'))
      outputs.append(elements.logger.JSONLOutput(
          logdir, 'scores.jsonl', 'episode/score'))
    elif output == 'tensorboard':
      outputs.append(elements.logger.TensorBoardOutput(
          logdir, config.logger.fps))
    elif output == 'expa':
      exp = logdir.split('/')[-4]
      run = '/'.join(logdir.split('/')[-3:])
      proj = 'embodied' if logdir.startswith(('/cns/', 'gs://')) else 'debug'
      outputs.append(elements.logger.ExpaOutput(
          exp, run, proj, config.logger.user, config.flat))
    elif output == 'wandb':
      name = '/'.join(logdir.split('/')[-4:])
      outputs.append(elements.logger.WandBOutput(name))
    elif output == 'scope':
      outputs.append(elements.logger.ScopeOutput(elements.Path(logdir)))
    else:
      raise NotImplementedError(output)
  logger = elements.Logger(step, outputs, multiplier)
  # 在每次add时保存log图片
  old_add = logger.add
  log_imgdir = os.path.join(logdir, 'images')
  def new_add(logs, *args, **kwargs):
    """Filter out very large / unsupported keys before handing to outputs.

    Some writers (for example ScopeOutput) do not accept large tensors or
    unfamiliar shapes. We make a best-effort filter here to drop keys that
    are known problematic (e.g. training random tensors) while still
    saving log images via save_log_images.
    """
    try:
      # Build a filtered copy for the writers.
      safe_logs = {}
      for k, v in logs.items():
        # Always allow scalar metrics and small arrays
        try:
          is_array = hasattr(v, 'ndim') and hasattr(v, 'shape')
        except Exception:
          is_array = False
        if is_array:
          # Drop known problematic namespaces (random tensors, full batches)
          if k.startswith('train/rand/') or k.startswith('train/batch/'):
            continue
          # Drop very large arrays to avoid writer format issues
          try:
            if getattr(v, 'size', 0) > 200000:  # ~200k elements
              continue
          except Exception:
            pass
        safe_logs[k] = v
      old_add(safe_logs, *args, **kwargs)
    except Exception as e:
      # Fall back to original behavior if filtering fails for any reason.
      try:
        old_add(logs, *args, **kwargs)
      except Exception:
        # If writers still fail, at least don't crash the whole training loop.
        print('[logger] Warning: writer failed even after fallback:', e)
    # Always try to save any image-like arrays from the original logs.
    try:
      save_log_images(logs, int(step), log_imgdir)
    except Exception as e:
      # Print error for debugging, but don't crash training
      print(f'[Save Log Image] Error: {e}')
  logger.add = new_add
  return logger


def make_replay(config, folder, mode='train'):
  batlen = config.batch_length if mode == 'train' else config.report_length
  consec = config.consec_train if mode == 'train' else config.consec_report
  capacity = config.replay.size if mode == 'train' else config.replay.size / 10
  length = consec * batlen + config.replay_context
  assert config.batch_size * length <= capacity

  directory = elements.Path(config.logdir) / folder
  if config.replicas > 1:
    directory /= f'{config.replica:05}'
  kwargs = dict(
      length=length, capacity=int(capacity), online=config.replay.online,
      chunksize=config.replay.chunksize, directory=directory)

  if config.replay.fracs.uniform < 1 and mode == 'train':
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    recency = 1.0 / np.arange(1, capacity + 1) ** config.replay.recexp
    selectors = embodied.replay.selectors
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)

  return embodied.replay.Replay(**kwargs)


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite == 'memmaze':
    from embodied.envs import from_gym
    import memory_maze  # noqa
  ctor = {
      'dummy': 'embodied.envs.dummy:Dummy',
      'gym': 'embodied.envs.from_gym:FromGym',
      'dm': 'embodied.envs.from_dmenv:FromDM',
      'crafter': 'embodied.envs.crafter:Crafter',
      'dmc': 'embodied.envs.dmc:DMC',
      'atari': 'embodied.envs.atari:Atari',
      'atari100k': 'embodied.envs.atari:Atari',
      'dmlab': 'embodied.envs.dmlab:DMLab',
      'minecraft': 'embodied.envs.minecraft:Minecraft',
      'loconav': 'embodied.envs.loconav:LocoNav',
      'pinpad': 'embodied.envs.pinpad:PinPad',
      'langroom': 'embodied.envs.langroom:LangRoom',
      'procgen': 'embodied.envs.procgen:ProcGen',
      'bsuite': 'embodied.envs.bsuite:BSuite',
      'metadrive': 'embodied.envs.metadrive_lane_keeping:MetaDriveLaneKeeping',
      'memmaze': lambda task, **kw: from_gym.FromGym(
          f'MemoryMaze-{task}-v0', **kw),
  }[suite]
  if isinstance(ctor, str):
    module, cls = ctor.split(':')
    module = importlib.import_module(module)
    ctor = getattr(module, cls)
  kwargs = config.env.get(suite, {})
  kwargs.update(overrides)
  # Allow forcing render from environment variable for quick visual debug.
  # Set DREAMER_USE_RENDER=1 or RENDER_AND_PRINT=1 to force use_render=True.
  if os.environ.get('DREAMER_USE_RENDER') == '1' or os.environ.get('RENDER_AND_PRINT') == '1':
    kwargs['use_render'] = True
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  if kwargs.pop('use_logdir', False):
    kwargs['logdir'] = elements.Path(config.logdir) / f'env{index}'
  env = ctor(task, **kwargs)
  return wrap_env(env, config)


def wrap_env(env, config):
  # Collect original action items first (avoid mutation while wrapping).
  act_items = list(env.act_space.items())

  # Normalize continuous actions (keeps them in [-1, 1] scale internally).
  for name, space in act_items:
    if not space.discrete:
      env = embodied.wrappers.NormalizeAction(env, name)

  # Perform space checks on the normalized space, but ensure values are
  # clipped before the checks run at step-time. To achieve that the ClipAction
  # must wrap CheckSpaces (i.e. be outer than it). So apply CheckSpaces first
  # and then wrap with ClipAction for every action key.
  env = embodied.wrappers.CheckSpaces(env)
  for name, space in act_items:
    if not space.discrete:
      env = embodied.wrappers.ClipAction(env, name)

  # Finally unify dtypes (outermost) so actions are converted to numpy dtypes
  # before being passed down the wrapper chain at runtime.
  env = embodied.wrappers.UnifyDtypes(env)
  return env


def make_stream(config, replay, mode):
  fn = bind(replay.sample, config.batch_size, mode)
  stream = embodied.streams.Stateless(fn)
  stream = embodied.streams.Consec(
      stream,
      length=config.batch_length if mode == 'train' else config.report_length,
      consec=config.consec_train if mode == 'train' else config.consec_report,
      prefix=config.replay_context,
      strict=(mode == 'train'),
      contiguous=True)

  return stream


if __name__ == '__main__':
  main()
