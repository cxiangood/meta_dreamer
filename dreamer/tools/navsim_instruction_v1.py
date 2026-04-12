"""
NAVSIM instruction-following v1 (offline behavior cloning).

Pipeline:
1) Load NAVSIM episodes from NPZ files.
2) Build pseudo-instruction labels from trajectory/state heuristics.
3) Train an instruction-conditioned policy to predict low-level actions.

Expected NPZ keys (minimal):
- image: uint8 [T, H, W, 3]
- action: float32 [T, 2]  (or steering + throttle_brake)

Optional keys used by pseudo labeling:
- speed, acceleration, current_steering, current_throttle_brake
- distance_to_route, route_completion
- angular_velocity: [T, 3]
- lead_distance, ttc, lane_change_hint
"""

import argparse
import json
import os
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization


INSTRUCTION_SET = [
    "keep lane",
    "center lane",
    "accelerate",
    "slow down",
    "brake",
    "change lane left",
    "change lane right",
]
INSTR2ID = {x: i for i, x in enumerate(INSTRUCTION_SET)}


def _safe_get(arrs: Dict[str, np.ndarray], key: str, length: int, default: float = 0.0):
    if key in arrs:
        x = np.asarray(arrs[key])
        if x.shape[0] == length:
            return x
    return np.full((length,), default, dtype=np.float32)


def pseudo_instruction(arrs: Dict[str, np.ndarray]) -> np.ndarray:
    length = arrs["image"].shape[0]
    speed = _safe_get(arrs, "speed", length, 0.0).astype(np.float32)
    dist_route = _safe_get(arrs, "distance_to_route", length, 0.0).astype(np.float32)
    lead_distance = _safe_get(arrs, "lead_distance", length, 50.0).astype(np.float32)
    ttc = _safe_get(arrs, "ttc", length, 99.0).astype(np.float32)
    lane_change_hint = _safe_get(arrs, "lane_change_hint", length, 0.0).astype(np.float32)

    labels = np.full((length,), INSTR2ID["keep lane"], dtype=np.int32)
    labels[np.abs(dist_route) > 0.7] = INSTR2ID["center lane"]
    labels[(speed < 2.5) & (lead_distance > 12.0)] = INSTR2ID["accelerate"]
    labels[(lead_distance < 15.0) | (ttc < 3.0)] = INSTR2ID["slow down"]
    labels[(lead_distance < 8.0) | (ttc < 1.5)] = INSTR2ID["brake"]
    labels[lane_change_hint < -0.5] = INSTR2ID["change lane left"]
    labels[lane_change_hint > 0.5] = INSTR2ID["change lane right"]
    return labels


def extract_state(arrs: Dict[str, np.ndarray]) -> np.ndarray:
    length = arrs["image"].shape[0]
    speed = _safe_get(arrs, "speed", length, 0.0).astype(np.float32)
    acc = _safe_get(arrs, "acceleration", length, 0.0).astype(np.float32)
    steer = _safe_get(arrs, "current_steering", length, 0.0).astype(np.float32)
    tb = _safe_get(arrs, "current_throttle_brake", length, 0.0).astype(np.float32)
    d2r = _safe_get(arrs, "distance_to_route", length, 0.0).astype(np.float32)
    rc = _safe_get(arrs, "route_completion", length, 0.0).astype(np.float32)
    ang = arrs.get("angular_velocity", np.zeros((length, 3), dtype=np.float32))
    ang = np.asarray(ang, dtype=np.float32)
    if ang.ndim != 2 or ang.shape[0] != length or ang.shape[1] != 3:
        ang = np.zeros((length, 3), dtype=np.float32)
    return np.concatenate(
        [speed[:, None], acc[:, None], steer[:, None], tb[:, None], d2r[:, None], rc[:, None], ang], axis=1
    ).astype(np.float32)


def parse_episode(npz_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(npz_path, allow_pickle=False)
    arrs = {k: np.asarray(raw[k]) for k in raw.files}
    if "image" not in arrs:
        raise ValueError(f"Missing key 'image' in {npz_path}")
    image = arrs["image"].astype(np.uint8)
    if image.ndim != 4 or image.shape[-1] != 3:
        raise ValueError(f"image must be [T,H,W,3], got {image.shape} in {npz_path}")
    length = image.shape[0]

    if "action" in arrs:
        action = np.asarray(arrs["action"], dtype=np.float32)
    elif "steering" in arrs and "throttle_brake" in arrs:
        action = np.stack([arrs["steering"], arrs["throttle_brake"]], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Need 'action' or ('steering','throttle_brake') in {npz_path}")
    if action.shape[0] != length or action.shape[-1] != 2:
        raise ValueError(f"action must be [T,2], got {action.shape} in {npz_path}")

    states = extract_state(arrs)
    instr = pseudo_instruction(arrs)
    return image, states, instr, action


def make_synthetic(n: int = 8192, h: int = 64, w: int = 64):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8)
    states = rng.normal(0, 1, size=(n, 9)).astype(np.float32)
    instr = rng.integers(0, len(INSTRUCTION_SET), size=(n,), dtype=np.int32)
    action = np.zeros((n, 2), dtype=np.float32)
    action[:, 0] = np.tanh(0.5 * states[:, 2] + (instr == INSTR2ID["change lane left"]) * -0.5 + (instr == INSTR2ID["change lane right"]) * 0.5)
    action[:, 1] = np.tanh(0.3 * states[:, 0] + (instr == INSTR2ID["accelerate"]) * 0.8 - (instr == INSTR2ID["brake"]) * 1.0)
    return image, states, instr, action


def load_dataset(data_dir: str, pattern: str, max_episodes: int):
    files = sorted(pathlib.Path(data_dir).rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found under {data_dir} with pattern {pattern}")
    if max_episodes > 0:
        files = files[:max_episodes]

    imgs, sts, ins, acts = [], [], [], []
    bad = 0
    for f in files:
        try:
            image, states, instr, action = parse_episode(f)
            imgs.append(image)
            sts.append(states)
            ins.append(instr)
            acts.append(action)
        except Exception:
            bad += 1
            continue
    if not imgs:
        raise RuntimeError("No valid episodes parsed from dataset.")
    image = np.concatenate(imgs, axis=0)
    states = np.concatenate(sts, axis=0)
    instr = np.concatenate(ins, axis=0)
    action = np.concatenate(acts, axis=0)
    return image, states, instr, action, len(files), bad


@dataclass
class Split:
    image: np.ndarray
    state: np.ndarray
    instr: np.ndarray
    action: np.ndarray


def train_val_split(image, state, instr, action, val_ratio=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = image.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1.0 - val_ratio))
    tr, va = idx[:cut], idx[cut:]
    return (
        Split(image[tr], state[tr], instr[tr], action[tr]),
        Split(image[va], state[va], instr[va], action[va]),
    )


class InstrPolicy(nn.Module):
    vocab_size: int
    state_dim: int = 9
    hidden: int = 256
    instr_dim: int = 64

    @nn.compact
    def __call__(self, image, state, instr):
        x = image.astype(jnp.float32) / 255.0
        x = nn.relu(nn.Conv(32, (8, 8), (4, 4), padding="SAME")(x))
        x = nn.relu(nn.Conv(64, (4, 4), (2, 2), padding="SAME")(x))
        x = nn.relu(nn.Conv(64, (3, 3), (1, 1), padding="SAME")(x))
        x = x.reshape((x.shape[0], -1))
        x = nn.tanh(nn.Dense(self.hidden)(x))

        s = nn.tanh(nn.Dense(128)(state))
        s = nn.tanh(nn.Dense(128)(s))

        emb = self.param("instr_emb", nn.initializers.normal(stddev=0.02), (self.vocab_size, self.instr_dim))
        t = emb[instr]
        t = nn.tanh(nn.Dense(128)(t))

        h = jnp.concatenate([x, s, t], axis=-1)
        h = nn.tanh(nn.Dense(self.hidden)(h))
        h = nn.tanh(nn.Dense(self.hidden)(h))
        return nn.Dense(2)(h)


def batch_iter(split: Split, batch_size: int, rng: np.random.Generator):
    n = split.image.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield (
            split.image[j],
            split.state[j],
            split.instr[j],
            split.action[j],
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic_steps", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="dreamer/logs_navsim/navsim_instruction_v1.params")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    np_rng = np.random.default_rng(args.seed)

    if args.synthetic:
        image, state, instr, action = make_synthetic(args.synthetic_steps)
        read_stats = {"files_seen": 0, "files_bad": 0}
    else:
        if not args.data_dir:
            raise ValueError("--data_dir is required unless --synthetic is set")
        image, state, instr, action, files_seen, files_bad = load_dataset(
            data_dir=args.data_dir, pattern=args.pattern, max_episodes=args.max_episodes
        )
        read_stats = {"files_seen": files_seen, "files_bad": files_bad}

    tr, va = train_val_split(image, state, instr, action, val_ratio=args.val_ratio, seed=args.seed)
    print(f"[Data] train={tr.image.shape[0]} val={va.image.shape[0]} dim_state={tr.state.shape[-1]} stats={read_stats}")
    hist = np.bincount(tr.instr, minlength=len(INSTRUCTION_SET))
    for i, name in enumerate(INSTRUCTION_SET):
        print(f"[Label] {name:>18}: {int(hist[i])}")

    model = InstrPolicy(vocab_size=len(INSTRUCTION_SET), state_dim=tr.state.shape[-1])
    key = jax.random.PRNGKey(args.seed)
    params = model.init(key, tr.image[:1], tr.state[:1], tr.instr[:1])["params"]
    tx = optax.adamw(learning_rate=args.lr, weight_decay=args.wd)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, image, state, instr, action):
        def loss_fn(p):
            pred = model.apply({"params": p}, image, state, instr)
            mse = jnp.mean((pred - action) ** 2)
            return mse

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_step(params, image, state, instr, action):
        pred = model.apply({"params": params}, image, state, instr)
        return jnp.mean((pred - action) ** 2)

    best_val = float("inf")
    best_params = params
    for ep in range(1, args.epochs + 1):
        train_losses = []
        for image_b, state_b, instr_b, action_b in batch_iter(tr, args.batch_size, np_rng):
            params, opt_state, loss = train_step(params, opt_state, image_b, state_b, instr_b, action_b)
            train_losses.append(float(loss))

        val_losses = []
        for image_b, state_b, instr_b, action_b in batch_iter(va, args.batch_size, np_rng):
            val_losses.append(float(eval_step(params, image_b, state_b, instr_b, action_b)))
        tr_mse = float(np.mean(train_losses)) if train_losses else 0.0
        va_mse = float(np.mean(val_losses)) if val_losses else tr_mse
        if va_mse < best_val:
            best_val = va_mse
            best_params = params
        print(f"[Epoch {ep:03d}] train_mse={tr_mse:.6f} val_mse={va_mse:.6f} best={best_val:.6f}")

    with open(args.save_path, "wb") as f:
        f.write(serialization.to_bytes(best_params))
    meta_path = args.save_path + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "instruction_set": INSTRUCTION_SET,
                "state_dim": int(tr.state.shape[-1]),
                "best_val_mse": best_val,
                "seed": args.seed,
                "read_stats": read_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[Done] Saved params: {args.save_path}")
    print(f"[Done] Saved meta:   {meta_path}")


if __name__ == "__main__":
    main()

