#!/usr/bin/env python3
"""Offline CFM pretraining on NAVSIM mini metadata.

This is a practical MVP for the first research step:

  NAVSIM logs -> future ego trajectories -> PCA latent -> conditional flow model

The learned checkpoint is intentionally simple and portable. It contains:
- trajectory/context normalization statistics
- PCA components that define the first latent space
- an MLP velocity field v_theta(x_tau, tau | context)

Later, the PCA latent can be replaced by Dreamer posterior latents without
changing the CFM training/evaluation protocol.
"""

import argparse
import json
import math
import os
import pathlib
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


EPS = 1e-6


@dataclass
class DatasetStats:
    files_seen: int
    files_used: int
    files_bad: int
    samples: int
    context_dim: int
    traj_dim: int
    horizon: int
    stride: int


def yaw_from_matrix(ego2global: np.ndarray) -> float:
    rot = np.asarray(ego2global, dtype=np.float64)[:2, :2]
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def frame_context(frame: Dict) -> np.ndarray:
    dyn = np.asarray(frame.get("ego_dynamic_state", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    if dyn.shape[0] < 4:
        dyn = np.pad(dyn, (0, 4 - dyn.shape[0]))
    dyn = dyn[:4]

    cmd = np.asarray(frame.get("driving_command", np.zeros(4)), dtype=np.float32)
    if cmd.shape[0] < 4:
        cmd = np.pad(cmd, (0, 4 - cmd.shape[0]))
    cmd = cmd[:4]

    can = np.asarray(frame.get("can_bus", np.zeros(18)), dtype=np.float32)
    if can.shape[0] < 18:
        can = np.pad(can, (0, 18 - can.shape[0]))
    # Skip global x/y/z and quaternion; retain local kinematic channels.
    can_tail = can[10:18]

    yaw = yaw_from_matrix(frame["ego2global"])
    return np.concatenate(
        [dyn, cmd, np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32), can_tail.astype(np.float32)],
        axis=0,
    ).astype(np.float32)


def future_trajectory(frames: List[Dict], idx: int, horizon: int, stride: int) -> np.ndarray:
    cur = frames[idx]
    origin = np.asarray(cur["ego2global_translation"], dtype=np.float64)[:2]
    rot = np.asarray(cur["ego2global"], dtype=np.float64)[:2, :2]
    local_from_global = rot.T
    points = []
    for h in range(1, horizon + 1):
        fut = frames[idx + h * stride]
        pos = np.asarray(fut["ego2global_translation"], dtype=np.float64)[:2]
        rel = local_from_global @ (pos - origin)
        points.append(rel.astype(np.float32))
    return np.concatenate(points, axis=0).astype(np.float32)


def iter_navsim_files(data_dir: str, pattern: str, max_files: int) -> List[pathlib.Path]:
    files = sorted(pathlib.Path(data_dir).rglob(pattern))
    if max_files > 0:
        files = files[:max_files]
    return files


def load_navsim_samples(
    data_dir: str,
    pattern: str,
    max_files: int,
    horizon: int,
    stride: int,
    sample_every: int,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, DatasetStats]:
    contexts, trajs = [], []
    files = iter_navsim_files(data_dir, pattern, max_files)
    bad = used = 0
    min_len = horizon * stride + 1
    for path in files:
        try:
            with open(path, "rb") as f:
                frames = pickle.load(f)
            if not isinstance(frames, list) or len(frames) < min_len:
                bad += 1
                continue
            added = 0
            stop = len(frames) - horizon * stride
            for idx in range(0, stop, sample_every):
                try:
                    contexts.append(frame_context(frames[idx]))
                    trajs.append(future_trajectory(frames, idx, horizon, stride))
                    added += 1
                except Exception:
                    continue
                if max_samples > 0 and len(trajs) >= max_samples:
                    break
            if added:
                used += 1
            if max_samples > 0 and len(trajs) >= max_samples:
                break
        except Exception:
            bad += 1
    if not trajs:
        raise RuntimeError(f"No valid NAVSIM samples found in {data_dir} with pattern {pattern}")
    contexts = np.asarray(contexts, dtype=np.float32)
    trajs = np.asarray(trajs, dtype=np.float32)
    stats = DatasetStats(
        files_seen=len(files),
        files_used=used,
        files_bad=bad,
        samples=int(trajs.shape[0]),
        context_dim=int(contexts.shape[1]),
        traj_dim=int(trajs.shape[1]),
        horizon=horizon,
        stride=stride,
    )
    return contexts, trajs, stats


def standardize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, EPS)
    return ((x - mean) / std).astype(np.float32), mean.squeeze(0), std.squeeze(0)


def fit_pca(x: np.ndarray, latent_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    # x should already be standardized and centered closely around zero.
    _, _, vt = np.linalg.svd(x.astype(np.float64), full_matrices=False)
    dim = min(latent_dim, vt.shape[0])
    comps = vt[:dim].astype(np.float32)
    latent = (x @ comps.T).astype(np.float32)
    return latent, comps


def split_arrays(*arrays: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    n = arrays[0].shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = max(1, int(n * (1.0 - val_ratio)))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return [(arr[train_idx], arr[val_idx]) for arr in arrays]


def init_mlp(rng: np.random.Generator, in_dim: int, hidden: int, out_dim: int) -> Dict[str, np.ndarray]:
    def w(shape):
        fan_in = shape[0]
        return (rng.normal(0, 1.0 / math.sqrt(fan_in), size=shape)).astype(np.float32)

    return {
        "W1": w((in_dim, hidden)),
        "b1": np.zeros((hidden,), dtype=np.float32),
        "W2": w((hidden, hidden)),
        "b2": np.zeros((hidden,), dtype=np.float32),
        "W3": w((hidden, out_dim)),
        "b3": np.zeros((out_dim,), dtype=np.float32),
    }


def mlp_forward(params: Dict[str, np.ndarray], x: np.ndarray):
    z1 = x @ params["W1"] + params["b1"]
    h1 = np.tanh(z1)
    z2 = h1 @ params["W2"] + params["b2"]
    h2 = np.tanh(z2)
    out = h2 @ params["W3"] + params["b3"]
    cache = (x, h1, h2)
    return out, cache


def mlp_loss_and_grads(params: Dict[str, np.ndarray], x: np.ndarray, target: np.ndarray):
    pred, (inp, h1, h2) = mlp_forward(params, x)
    diff = pred - target
    loss = float(np.mean(diff * diff))
    g = (2.0 / diff.size) * diff

    grads = {}
    grads["W3"] = h2.T @ g
    grads["b3"] = g.sum(axis=0)
    gh2 = g @ params["W3"].T
    gz2 = gh2 * (1.0 - h2 * h2)
    grads["W2"] = h1.T @ gz2
    grads["b2"] = gz2.sum(axis=0)
    gh1 = gz2 @ params["W2"].T
    gz1 = gh1 * (1.0 - h1 * h1)
    grads["W1"] = inp.T @ gz1
    grads["b1"] = gz1.sum(axis=0)
    return loss, {k: v.astype(np.float32) for k, v in grads.items()}


def adam_update(params, grads, opt, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
    opt["t"] += 1
    t = opt["t"]
    for k, p in params.items():
        g = grads[k]
        if weight_decay:
            g = g + weight_decay * p
        opt["m"][k] = beta1 * opt["m"][k] + (1.0 - beta1) * g
        opt["v"][k] = beta2 * opt["v"][k] + (1.0 - beta2) * (g * g)
        mh = opt["m"][k] / (1.0 - beta1**t)
        vh = opt["v"][k] / (1.0 - beta2**t)
        params[k] = (p - lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)


def make_cfm_batch(context, latent, batch_idx, rng):
    c = context[batch_idx]
    z1 = latent[batch_idx]
    z0 = rng.normal(size=z1.shape).astype(np.float32)
    tau = rng.uniform(0.0, 1.0, size=(z1.shape[0], 1)).astype(np.float32)
    xt = (1.0 - tau) * z0 + tau * z1
    target_v = z1 - z0
    inp = np.concatenate([xt, tau, c], axis=1).astype(np.float32)
    return inp, target_v


def eval_cfm(params, context, latent, batch_size, rng) -> float:
    losses = []
    n = latent.shape[0]
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        inp, target = make_cfm_batch(context, latent, idx, rng)
        pred, _ = mlp_forward(params, inp)
        losses.append(float(np.mean((pred - target) ** 2)))
    return float(np.mean(losses)) if losses else 0.0


def save_checkpoint(path: str, params: Dict[str, np.ndarray], arrays: Dict[str, np.ndarray], metadata: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {**params, **arrays, "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False))}
    np.savez(path, **payload)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def train_torch(
    args,
    outdir: pathlib.Path,
    context_train: np.ndarray,
    latent_train: np.ndarray,
    context_val: np.ndarray,
    latent_val: np.ndarray,
    stats_arrays: Dict[str, np.ndarray],
    metadata: Dict,
) -> str:
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = nn.Sequential(
        nn.Linear(args.latent_dim + 1 + context_train.shape[1], args.hidden),
        nn.Tanh(),
        nn.Linear(args.hidden, args.hidden),
        nn.Tanh(),
        nn.Linear(args.hidden, args.latent_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ctx_tr = torch.as_tensor(context_train, dtype=torch.float32, device=device)
    lat_tr = torch.as_tensor(latent_train, dtype=torch.float32, device=device)
    ctx_va = torch.as_tensor(context_val, dtype=torch.float32, device=device)
    lat_va = torch.as_tensor(latent_val, dtype=torch.float32, device=device)

    def make_batch(ctx, lat, idx):
        c = ctx[idx]
        z1 = lat[idx]
        z0 = torch.randn_like(z1)
        tau = torch.rand((z1.shape[0], 1), device=device)
        xt = (1.0 - tau) * z0 + tau * z1
        target = z1 - z0
        inp = torch.cat([xt, tau, c], dim=-1)
        return inp, target

    @torch.no_grad()
    def eval_loss():
        model.eval()
        losses = []
        n = lat_va.shape[0]
        for start in range(0, n, args.batch_size):
            idx = torch.arange(start, min(start + args.batch_size, n), device=device)
            inp, target = make_batch(ctx_va, lat_va, idx)
            pred = model(inp)
            losses.append(torch.mean((pred - target) ** 2).item())
        model.train()
        return float(np.mean(losses)) if losses else 0.0

    best_val = float("inf")
    best_path = str(outdir / "best_ckpt.pt")
    n = lat_tr.shape[0]
    print(f"[Torch] device={device} train={n} val={lat_va.shape[0]}")
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []
        model.train()
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            inp, target = make_batch(ctx_tr, lat_tr, idx)
            pred = model(inp)
            loss = torch.mean((pred - target) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_loss = eval_loss()
        print(f"[Epoch {epoch:04d}] train_cfm={train_loss:.6f} val_cfm={val_loss:.6f} best={best_val:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            metadata["best_val_cfm"] = best_val
            metadata["best_epoch"] = epoch
            metadata["backend"] = "torch"
            metadata["device"] = str(device)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metadata": metadata,
                    "stats": stats_arrays,
                },
                best_path,
            )
            with open(best_path + ".json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"[Save] {best_path}")
        elif epoch % args.save_every == 0 or epoch == args.epochs:
            path = str(outdir / f"ckpt_epoch_{epoch:04d}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "metadata": metadata,
                    "stats": stats_arrays,
                },
                path,
            )
            print(f"[Save] {path}")

    print(f"[Done] best_val_cfm={best_val:.6f}")
    print(f"[Done] best_ckpt={best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--outdir", default="dreamer/logs_navsim_cfm")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    run_name = args.run_name or time.strftime("navsim_cfm_%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] data_dir={args.data_dir} pattern={args.pattern}")
    context, traj, data_stats = load_navsim_samples(
        args.data_dir,
        args.pattern,
        args.max_files,
        args.horizon,
        args.stride,
        args.sample_every,
        args.max_samples,
    )
    print(f"[Data] {asdict(data_stats)}")

    context_norm, context_mean, context_std = standardize(context)
    traj_norm, traj_mean, traj_std = standardize(traj)
    latent, pca_components = fit_pca(traj_norm, args.latent_dim)
    latent_norm, latent_mean, latent_std = standardize(latent)
    if latent_norm.shape[1] != args.latent_dim:
        print(f"[Warn] latent_dim clamped from {args.latent_dim} to {latent_norm.shape[1]} by PCA rank.")
        args.latent_dim = int(latent_norm.shape[1])

    (ctx_tr, ctx_va), (lat_tr, lat_va) = split_arrays(
        context_norm, latent_norm, val_ratio=args.val_ratio, seed=args.seed
    )

    in_dim = args.latent_dim + 1 + context_norm.shape[1]
    params = init_mlp(rng, in_dim, args.hidden, args.latent_dim)
    opt = {
        "t": 0,
        "m": {k: np.zeros_like(v) for k, v in params.items()},
        "v": {k: np.zeros_like(v) for k, v in params.items()},
    }

    metadata = {
        "method": "pca_latent_conditional_flow_matching",
        "note": "MVP latent is PCA over normalized NAVSIM future ego trajectories.",
        "args": vars(args),
        "dataset": asdict(data_stats),
        "input_dim": int(in_dim),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    stats_arrays = {
        "context_mean": context_mean,
        "context_std": context_std,
        "traj_mean": traj_mean,
        "traj_std": traj_std,
        "pca_components": pca_components,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
    }

    use_torch = args.backend in ("auto", "torch")
    if use_torch:
        try:
            train_torch(args, outdir, ctx_tr, lat_tr, ctx_va, lat_va, stats_arrays, metadata)
            return
        except ImportError as err:
            if args.backend == "torch":
                raise
            print(f"[Warn] torch unavailable ({err}); falling back to NumPy backend.")

    best_val = float("inf")
    best_path = ""
    n = lat_tr.shape[0]
    print(f"[Train] train={n} val={lat_va.shape[0]} in_dim={in_dim} latent_dim={args.latent_dim}")
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            inp, target = make_cfm_batch(ctx_tr, lat_tr, idx, rng)
            loss, grads = mlp_loss_and_grads(params, inp, target)
            adam_update(params, grads, opt, args.lr, weight_decay=args.weight_decay)
            losses.append(loss)

        train_loss = float(np.mean(losses))
        val_loss = eval_cfm(params, ctx_va, lat_va, args.batch_size, np.random.default_rng(args.seed + epoch))
        print(f"[Epoch {epoch:04d}] train_cfm={train_loss:.6f} val_cfm={val_loss:.6f} best={best_val:.6f}")

        should_save = epoch == args.epochs or epoch % args.save_every == 0 or val_loss < best_val
        if val_loss < best_val:
            best_val = val_loss
            metadata["best_val_cfm"] = best_val
            metadata["best_epoch"] = epoch
            best_path = str(outdir / "best_ckpt.npz")
            should_save = True

        if should_save:
            ckpt_path = str(outdir / ("best_ckpt.npz" if val_loss <= best_val else f"ckpt_epoch_{epoch:04d}.npz"))
            metadata["backend"] = "numpy"
            save_checkpoint(ckpt_path, params, stats_arrays, metadata)
            print(f"[Save] {ckpt_path}")

    print(f"[Done] best_val_cfm={best_val:.6f}")
    print(f"[Done] best_ckpt={best_path or str(outdir / 'best_ckpt.npz')}")


if __name__ == "__main__":
    main()
