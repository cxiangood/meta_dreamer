import argparse
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
import optax

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.policy.idm_policy import IDMPolicy

import embodied
from embodied.agents.actor_critic_jax import ActorCriticAgent
from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping


# --------- Common helpers ---------

def pack_state(obs: Dict) -> np.ndarray:
    """Pack scalar features to match train_actor_critic format."""
    ang = np.asarray(obs.get("angular_velocity", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)
    vec = np.array([
        obs.get("speed", 0.0),
        obs.get("acceleration", 0.0),
        obs.get("current_steering", 0.0),
        obs.get("current_throttle_brake", 0.0),
        obs.get("distance_to_route", 0.0),
        obs.get("route_completion", 0.0),
    ], dtype=np.float32)
    return np.concatenate([vec, ang]).astype(np.float32)


def ensure_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


class ILThenRL(embodied.Env):
    """Thin wrapper so IL→RL can be used like other embodied envs."""

    def __init__(self, task=None, size=(64, 64), repeat=1, length=1000, **kwargs):
        # Delegate all environment logic to the MetaDrive lane-keeping wrapper.
        self._env = MetaDriveLaneKeeping(task, size=size, repeat=repeat, length=length, **kwargs)

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env._reset()

    def _reset(self):
        return self._env._reset()

    def close(self):
        self._env.close()


# --------- IL data collection (IDM expert) ---------

def make_idm_env(image_size=(64, 64), traffic_density=0.0, map_id=4):
    cfg = dict(
        use_render=False,
        manual_control=False,
        traffic_density=traffic_density,
        num_scenarios=200,
        random_agent_model=False,
        random_lane_width=False,
        random_lane_num=False,
        random_spawn_lane_index=False,
        on_continuous_line_done=False,
        out_of_route_done=True,
        map=map_id,
        start_seed=0,
        image_observation=True,
        sensors=dict(rgb_camera=(RGBCamera, image_size[0], image_size[1])),
        interface_panel=["rgb_camera", "dashboard"],
        agent_policy=IDMPolicy,
    )
    return MetaDriveEnv(cfg)


def collect_idm_dataset(
    episodes: int,
    max_steps: int,
    out_path: Path,
    image_size=(64, 64),
    traffic_density=0.0,
    map_id=4,
):
    env = make_idm_env(image_size=image_size, traffic_density=traffic_density, map_id=map_id)
    obs_buf, state_buf, act_buf = [], [], []
    ep_returns = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            ep_ret = 0.0
            for _ in range(max_steps):
                if isinstance(obs, dict):
                    img = obs.get("image")
                else:
                    break
                if img is None:
                    break
                action = env.agent.policy.action_info.get("action") if hasattr(env.agent, "policy") else None
                if action is None:
                    action = env.agent.policy.act(env.agent.id)
                action = np.array(action, dtype=np.float32)
                throttle_brake = np.clip(action[1] / 4.0, -1.0, 1.0)
                obs_buf.append(img.astype(np.uint8))
                svec = pack_state(dict(speed=0.0, acceleration=0.0, current_steering=action[0], current_throttle_brake=throttle_brake, distance_to_route=0.0, route_completion=0.0, angular_velocity=[0, 0, 0]))
                state_buf.append(svec)
                act_buf.append(np.array([action[0], throttle_brake], dtype=np.float32))
                obs, r, tm, tc, info = env.step(action)
                ep_ret += float(r)
                if tm or tc:
                    break
            ep_returns.append(ep_ret)
            print(f"[Collect] ep {ep+1}/{episodes} return {ep_ret:.2f} steps {len(obs_buf)}")
    finally:
        env.close()
    obs_arr = np.stack(obs_buf)
    state_arr = np.stack(state_buf)
    act_arr = np.stack(act_buf)
    ensure_dir(out_path)
    np.savez_compressed(out_path, images=obs_arr, states=state_arr, actions=act_arr, ep_returns=np.array(ep_returns))
    print(f"[Collect] saved {len(obs_arr)} steps to {out_path}")
    return out_path


# --------- IL pretraining (supervised on expert actions) ---------

def il_pretrain(dataset_path: Path, epochs: int, batch_size: int, lr: float, seed: int = 0):
    data = np.load(dataset_path)
    images = data["images"]
    states = data["states"]
    actions = data["actions"]
    n = images.shape[0]
    rng = jax.random.PRNGKey(seed)
    agent = ActorCriticAgent(rng, state_dim=states.shape[1])

    opt = optax.adam(lr)
    opt_state = opt.init(agent.params)

    def loss_fn(params, img_batch, state_batch, act_batch):
        feats = jax.vmap(lambda im, st: agent.encoder.apply({"params": params["encoder"]}, im, st))(img_batch, state_batch)
        mus, stds = jax.vmap(lambda f: agent.actor.apply({"params": params["actor"]}, f))(feats)
        var = stds ** 2
        logp = -0.5 * jnp.sum(((act_batch - mus) ** 2) / (var + 1e-8) + 2 * jnp.log(stds + 1e-8) + jnp.log(2 * jnp.pi), axis=-1)
        nll = -jnp.mean(logp)
        return nll

    @jax.jit
    def train_step(params, opt_state, img_batch, state_batch, act_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, img_batch, state_batch, act_batch)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    steps_per_epoch = max(1, n // batch_size)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        losses = []
        for i in range(steps_per_epoch):
            sel = idx[i * batch_size:(i + 1) * batch_size]
            if len(sel) == 0:
                continue
            params, opt_state, loss = train_step(
                agent.params,
                opt_state,
                jnp.array(images[sel]),
                jnp.array(states[sel]),
                jnp.array(actions[sel]),
            )
            agent.params = params
            losses.append(float(loss))
        mean_loss = np.mean(losses) if losses else 0.0
        print(f"[IL] epoch {ep+1}/{epochs} loss {mean_loss:.4f}")
    return agent


# --------- RL fine-tune (uses existing actor_critic_jax update) ---------

def simple_rollout(env, agent, steps_per_rollout=128):
    obs = env._reset()
    images = []
    states = []
    actions = []
    rewards = []
    dones = []
    total_reward = 0.0

    for _ in range(steps_per_rollout):
        img = obs['image']
        state_vec = pack_state(obs)
        act = agent.select_action(img, state_vec)
        action = {'steering': float(act[0]), 'throttle_brake': float(act[1]), 'reset': False}
        obs = env.step(action)
        r = float(obs['reward'])
        done = bool(obs['is_last'])
        images.append(img.astype(np.uint8))
        states.append(state_vec.astype(np.float32))
        actions.append(act.astype(np.float32))
        rewards.append(r)
        dones.append(done)
        total_reward += r
        if done:
            obs = env._reset()
            break
    batch = dict(
        images=np.stack(images),
        states=np.stack(states),
        actions=np.stack(actions),
        rewards=np.array(rewards),
        dones=np.array(dones),
    )
    return batch, total_reward, done, obs


def rl_finetune(agent, num_iterations=500, steps_per_rollout=128):
    env = MetaDriveLaneKeeping(task=None)
    for it in range(num_iterations):
        batch, tot_r, done, _ = simple_rollout(env, agent, steps_per_rollout)
        if batch['images'].shape[0] == 0:
            continue
        try:
            import jax.numpy as jnp
            last_feat = agent._encode(
                agent.params,
                jnp.expand_dims(batch['images'][-1], 0),
                jnp.expand_dims(batch['states'][-1], 0),
            )
            last_value = float(agent.critic.apply({'params': agent.params['critic']}, last_feat))
        except Exception:
            last_value = 0.0
        returns = agent._compute_returns(batch['rewards'], batch['dones'], last_value)
        try:
            feats = jax.vmap(
                lambda im, st: agent.encoder.apply({'params': agent.params['encoder']}, im, st)
            )(batch['images'], batch['states'])
            values = jax.vmap(lambda f: agent.critic.apply({'params': agent.params['critic']}, f))(feats)
            advantages = returns - np.array(values)
        except Exception:
            advantages = returns
        train_batch = dict(
            images=batch['images'],
            states=batch['states'],
            actions=batch['actions'],
            returns=returns,
            advantages=advantages,
        )
        loss, aux = agent.update(train_batch)
        if it % 10 == 0:
            lr_show = aux.get('lr', agent.get_lr()) if isinstance(aux, dict) else agent.get_lr()
            print(f"[RL] iter {it:04d} loss {loss:.4f} pol {aux['policy_loss']:.4f} val {aux['value_loss']:.4f} ent {aux['entropy']:.4f} epR {tot_r:.2f} lr {lr_show:.2e}")
    env.close()


# --------- CLI ---------

def main():
    parser = argparse.ArgumentParser(description="IL (IDM expert) then RL fine-tune")
    parser.add_argument('--collect_episodes', type=int, default=50)
    parser.add_argument('--collect_steps', type=int, default=500)
    parser.add_argument('--dataset', type=str, default='logs/il/idm_dataset.npz')
    parser.add_argument('--il_epochs', type=int, default=5)
    parser.add_argument('--il_batch', type=int, default=256)
    parser.add_argument('--il_lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip_collect', action='store_true')
    parser.add_argument('--skip_il', action='store_true')
    parser.add_argument('--rl_iters', type=int, default=300)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not args.skip_collect:
        collect_idm_dataset(args.collect_episodes, args.collect_steps, dataset_path)
    else:
        print(f"[Skip] data collection, using {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset {dataset_path} not found")

    if args.skip_il:
        rng = jax.random.PRNGKey(args.seed)
        agent = ActorCriticAgent(rng, state_dim=9)
    else:
        agent = il_pretrain(dataset_path, args.il_epochs, args.il_batch, args.il_lr, seed=args.seed)

    rl_finetune(agent, num_iterations=args.rl_iters)


if __name__ == '__main__':
    main()
