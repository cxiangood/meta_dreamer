import os
import time
import numpy as np
from embodied.envs.metadrive_lane_keeping import MetaDriveLaneKeeping
from embodied.agents.actor_critic_jax import ActorCriticAgent
import jax

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

def simple_rollout(env, agent, steps_per_rollout=128):
    obs = env._reset()
    images = []
    states = []
    actions = []
    rewards = []
    dones = []
    total_reward = 0.0

    def pack_state(o):
        ang = np.asarray(o['angular_velocity'], dtype=np.float32).reshape(-1)
        vec = np.array([
            o['speed'],
            o['acceleration'],
            o['current_steering'],
            o['current_throttle_brake'],
            o['distance_to_route'],
            o['route_completion'],
        ], dtype=np.float32)
        return np.concatenate([vec, ang]).astype(np.float32)

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


def train(num_iterations=1000, steps_per_rollout=128):
    env = MetaDriveLaneKeeping(task=None)
    rng = jax.random.PRNGKey(int(time.time()) & 0xffffffff)
    agent = ActorCriticAgent(rng, state_dim=9)

    writer = None
    logdir = os.environ.get('TB_LOGDIR', os.path.join(os.getcwd(), 'logs', 'tb'))
    if SummaryWriter is not None:
        try:
            writer = SummaryWriter(logdir)
            print(f"[TB] Logging to {logdir}")
        except Exception as e:
            print(f"[TB] Failed to create SummaryWriter: {e}")

    for it in range(num_iterations):
        batch, tot_r, done, _ = simple_rollout(env, agent, steps_per_rollout)
        if batch['images'].shape[0] == 0:
            continue
        # compute last value for bootstrap
        last_img = jnp = None
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
        # simple advantages
        # compute values for all states
        try:
            import jax.numpy as jnp
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

        if writer is not None:
            try:
                writer.add_scalar('loss/total', float(loss), agent.train_steps)
                writer.add_scalar('loss/policy', float(aux['policy_loss']), agent.train_steps)
                writer.add_scalar('loss/value', float(aux['value_loss']), agent.train_steps)
                writer.add_scalar('entropy', float(aux['entropy']), agent.train_steps)
                writer.add_scalar('reward/episode', float(tot_r), agent.train_steps)
                if 'lr' in aux:
                    writer.add_scalar('lr', float(aux['lr']), agent.train_steps)
            except Exception as e:
                print(f"[TB] log failed: {e}")

        if it % 10 == 0:
            lr_show = aux.get('lr', agent.get_lr()) if isinstance(aux, dict) else agent.get_lr()
            print(f"Iter {it:04d} Loss {loss:.4f} Policy {aux['policy_loss']:.4f} Value {aux['value_loss']:.4f} Ent {aux['entropy']:.4f} TotR {tot_r:.2f} LR {lr_show:.2e}")

    if writer is not None:
        writer.close()

if __name__ == '__main__':
    train(500, 128)
