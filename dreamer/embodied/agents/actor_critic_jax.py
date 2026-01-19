import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from typing import Sequence, Tuple

# Simple CNN encoder for 64x64x3 images with optional state branch
class Encoder(nn.Module):
    latent_dim: int = 256
    state_dim: int = 0

    @nn.compact
    def __call__(self, image, state=None):
        # image: uint8 [H,W,C] or [B,H,W,C]
        x = image.astype(jnp.float32) / 255.0
        x = nn.Conv(32, (8, 8), strides=(4, 4), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(64, (4, 4), strides=(2, 2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(1, 1), padding='SAME')(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1)) if x.ndim == 4 else x.reshape(-1)
        img_feat = nn.Dense(self.latent_dim)(x)
        img_feat = nn.tanh(img_feat)

        if self.state_dim > 0 and state is not None:
            st = state.astype(jnp.float32)
            st = nn.Dense(128)(st)
            st = nn.tanh(st)
            st = nn.Dense(128)(st)
            st = nn.tanh(st)
            feat = jnp.concatenate([img_feat, st], axis=-1)
            feat = nn.Dense(self.latent_dim)(feat)
            feat = nn.tanh(feat)
            return feat

        return img_feat

class Actor(nn.Module):
    action_dim: int = 2
    hidden: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, feat):
        x = feat
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        mu = nn.Dense(self.action_dim)(x)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        std = jnp.exp(log_std)
        return mu, std

class Critic(nn.Module):
    hidden: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, feat):
        x = feat
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.tanh(x)
        v = nn.Dense(1)(x)
        return jnp.squeeze(v, -1)

class ActorCriticAgent:
    def __init__(
        self,
        rng: jax.random.PRNGKey,
        action_dim=2,
        lr=3e-4,
        lr_warmup_steps: int = 10000,
        lr_decay_steps: int = 500000,
        lr_min_ratio: float = 0.1,
        state_dim: int = 0,
    ):
        self.rng = rng
        self.state_dim = int(state_dim)
        self.use_state = self.state_dim > 0
        self.encoder = Encoder(state_dim=self.state_dim)
        self.actor = Actor(action_dim)
        self.critic = Critic()

        dummy_img = jnp.zeros((1, 64, 64, 3), dtype=jnp.uint8)
        dummy_state = jnp.zeros((1, self.state_dim), dtype=jnp.float32) if self.use_state else None
        enc_vars = self.encoder.init(self.rng, dummy_img, dummy_state)
        feat = self.encoder.apply(enc_vars, dummy_img, dummy_state)
        actor_vars = self.actor.init(self.rng, feat)
        critic_vars = self.critic.init(self.rng, feat)

        # Pack params
        params = {
            'encoder': enc_vars['params'],
            'actor': actor_vars['params'],
            'critic': critic_vars['params']
        }
        self.params = params

        # Learning rate schedule: linear warmup -> cosine decay
        warm_steps = max(1, int(lr_warmup_steps))
        decay_steps = max(1, int(lr_decay_steps))
        warmup = optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warm_steps)
        cosine = optax.cosine_decay_schedule(init_value=lr, decay_steps=decay_steps, alpha=lr_min_ratio)
        lr_schedule = optax.join_schedules(schedules=[warmup, cosine], boundaries=[warm_steps])

        optimizer = optax.adam(lr_schedule)
        self.opt_state = optimizer.init(self.params)
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.train_steps = 0

        # jit compiled functions
        self._policy_apply = jax.jit(self._policy_apply_fn)
        self._loss_and_grads = jax.jit(jax.value_and_grad(self._loss_fn, argnums=0, has_aux=True))

    def _encode(self, params, obs_image, obs_state=None):
        return self.encoder.apply({'params': params['encoder']}, obs_image, obs_state)

    def _policy_apply_fn(self, params, obs_image, obs_state, rng):
        feat = self._encode(params, obs_image, obs_state)
        mu, std = self.actor.apply({'params': params['actor']}, feat)
        key, sub = jax.random.split(rng)
        eps = jax.random.normal(sub, mu.shape)
        action = mu + eps * std
        return action, mu, std, key

    def select_action(self, obs_image: np.ndarray, obs_state: np.ndarray = None) -> Tuple[np.ndarray, jax.random.PRNGKey]:
        # obs_image: HWC uint8; obs_state: vector features or None
        img = jnp.expand_dims(jnp.array(obs_image), 0)
        state = None
        if self.use_state:
            if obs_state is None:
                state = jnp.zeros((1, self.state_dim), dtype=jnp.float32)
            else:
                state = jnp.expand_dims(jnp.array(obs_state, dtype=jnp.float32), 0)
        action, mu, std, self.rng = self._policy_apply(self.params, img, state, self.rng)
        return np.asarray(action[0])

    def _compute_returns(self, rewards, dones, last_value, gamma=0.99):
        returns = np.zeros_like(rewards)
        running = last_value
        for t in reversed(range(len(rewards))):
            running = rewards[t] + gamma * running * (1.0 - dones[t])
            returns[t] = running
        return returns

    def _loss_fn(self, params, batch):
        # batch contains: images [T,H,W,C], actions [T,2], returns [T], advantages [T]
        imgs = batch['images']
        acts = batch['actions']
        rets = batch['returns']
        advs = batch['advantages']
        states = batch.get('states', None)

        if self.use_state and states is not None:
            feats = jax.vmap(lambda im, st: self.encoder.apply({'params': params['encoder']}, im, st))(imgs, states)
        else:
            feats = jax.vmap(lambda im: self.encoder.apply({'params': params['encoder']}, im, None))(imgs)
        mus, stds = jax.vmap(lambda f: self.actor.apply({'params': params['actor']}, f))(feats)
        vals = jax.vmap(lambda f: self.critic.apply({'params': params['critic']}, f))(feats)

        # policy log prob under Gaussian
        var = stds ** 2
        logp = -0.5 * jnp.sum(((acts - mus) ** 2) / (var + 1e-8) + 2 * jnp.log(stds + 1e-8) + jnp.log(2 * jnp.pi), axis=-1)
        policy_loss = -jnp.mean(logp * advs)
        value_loss = jnp.mean((rets - vals) ** 2)
        entropy = jnp.mean(jnp.sum(0.5 * (jnp.log(2 * jnp.pi * var) + 1.0), axis=-1))

        loss = policy_loss + 0.5 * value_loss - 1e-3 * entropy
        aux = dict(policy_loss=policy_loss, value_loss=value_loss, entropy=entropy, mean_value=jnp.mean(vals))
        return loss, aux

    def update(self, batch):
        (loss, aux), grads = self._loss_and_grads(self.params, batch)
        current_lr = float(self.lr_schedule(self.train_steps)) if hasattr(self, 'lr_schedule') else None
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
        self.params = optax.apply_updates(self.params, updates)
        self.train_steps += 1
        if current_lr is not None:
            aux = dict(aux, lr=current_lr)
        return loss, aux

    def get_lr(self) -> float:
        """Return current learning rate based on internal step counter."""
        return float(self.lr_schedule(self.train_steps)) if hasattr(self, 'lr_schedule') else 0.0
