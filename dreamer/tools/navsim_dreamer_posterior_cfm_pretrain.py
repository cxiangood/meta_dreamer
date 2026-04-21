#!/usr/bin/env python3
"""NAVSIM offline pretraining with Dreamer posterior latents + CFM.

Pipeline:
1) Load NAVSIM mini metadata pkl files.
2) Train a compact Dreamer-style RSSM world model on ego/context sequences.
3) Extract posterior latents z_t from the trained world model.
4) Train Conditional Flow Matching to model future posterior latent sequences.

This is intentionally a PyTorch offline pretraining entrypoint. It produces a
single checkpoint containing the RSSM posterior world model, CFM velocity
network, normalizers, and training metadata.
"""

import argparse
import json
import math
import os
import pathlib
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


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
    can_tail = can[10:18]

    yaw = yaw_from_matrix(frame["ego2global"])
    return np.concatenate(
        [dyn, cmd, np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32), can_tail.astype(np.float32)],
        axis=0,
    ).astype(np.float32)


@dataclass
class DatasetStats:
    files_seen: int
    files_used: int
    files_bad: int
    sequences: int
    obs_dim: int
    context_len: int
    horizon: int
    sample_every: int


def load_sequences(
    data_dir: str,
    pattern: str,
    max_files: int,
    context_len: int,
    horizon: int,
    sample_every: int,
    max_sequences: int,
) -> Tuple[np.ndarray, DatasetStats]:
    files = sorted(pathlib.Path(data_dir).rglob(pattern))
    if max_files > 0:
        files = files[:max_files]
    seq_len = context_len + horizon
    seqs = []
    used = bad = 0
    for path in files:
        try:
            with open(path, "rb") as f:
                frames = pickle.load(f)
            if not isinstance(frames, list) or len(frames) < seq_len:
                bad += 1
                continue
            arr = np.stack([frame_context(x) for x in frames], axis=0).astype(np.float32)
            added = 0
            for start in range(0, len(arr) - seq_len + 1, sample_every):
                seqs.append(arr[start : start + seq_len])
                added += 1
                if max_sequences > 0 and len(seqs) >= max_sequences:
                    break
            if added:
                used += 1
            if max_sequences > 0 and len(seqs) >= max_sequences:
                break
        except Exception:
            bad += 1
    if not seqs:
        raise RuntimeError(f"No valid NAVSIM sequences found under {data_dir}")
    data = np.asarray(seqs, dtype=np.float32)
    stats = DatasetStats(
        files_seen=len(files),
        files_used=used,
        files_bad=bad,
        sequences=int(data.shape[0]),
        obs_dim=int(data.shape[-1]),
        context_len=context_len,
        horizon=horizon,
        sample_every=sample_every,
    )
    return data, stats


def normalize(data: np.ndarray):
    flat = data.reshape((-1, data.shape[-1]))
    mean = flat.mean(axis=0).astype(np.float32)
    std = np.maximum(flat.std(axis=0).astype(np.float32), 1e-6)
    return ((data - mean) / std).astype(np.float32), mean, std


def split_data(data: np.ndarray, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(data.shape[0])
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * (1.0 - val_ratio)))
    return data[idx[:cut]], data[idx[cut:]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--outdir", default="dreamer/logs_navsim_cfm")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--context_len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--deter_dim", type=int, default=256)
    parser.add_argument("--stoch_dim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--world_epochs", type=int, default=80)
    parser.add_argument("--cfm_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--free_nats", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    run_name = args.run_name or time.strftime("navsim_dreamer_posterior_cfm_%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] data_dir={args.data_dir} pattern={args.pattern}")
    data, stats = load_sequences(
        args.data_dir,
        args.pattern,
        args.max_files,
        args.context_len,
        args.horizon,
        args.sample_every,
        args.max_sequences,
    )
    data, obs_mean, obs_std = normalize(data)
    train_np, val_np = split_data(data, args.val_ratio, args.seed)
    print(f"[Data] {asdict(stats)} train={len(train_np)} val={len(val_np)}")

    obs_dim = stats.obs_dim
    feat_dim = args.deter_dim + args.stoch_dim
    future_latent_dim = args.horizon * args.stoch_dim

    class RSSMWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(obs_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, args.hidden), nn.ELU())
            self.gru = nn.GRUCell(args.stoch_dim, args.deter_dim)
            self.post = nn.Linear(args.deter_dim + args.hidden, 2 * args.stoch_dim)
            self.prior = nn.Sequential(nn.Linear(args.deter_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 2 * args.stoch_dim))
            self.dec = nn.Sequential(nn.Linear(feat_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, args.hidden), nn.ELU(), nn.Linear(args.hidden, obs_dim))

        def dist(self, stats):
            mean, rawstd = stats.chunk(2, dim=-1)
            std = F.softplus(rawstd) + 0.1
            return mean, std

        def forward(self, obs):
            # obs: [B, T, D]
            b, t, _ = obs.shape
            h = torch.zeros((b, args.deter_dim), device=obs.device)
            z = torch.zeros((b, args.stoch_dim), device=obs.device)
            posts, priors, zs, hs, recons = [], [], [], [], []
            for i in range(t):
                h = self.gru(z, h)
                token = self.enc(obs[:, i])
                prior_m, prior_s = self.dist(self.prior(h))
                post_m, post_s = self.dist(self.post(torch.cat([h, token], dim=-1)))
                eps = torch.randn_like(post_m)
                z = post_m + eps * post_s
                recon = self.dec(torch.cat([h, z], dim=-1))
                posts.append((post_m, post_s))
                priors.append((prior_m, prior_s))
                zs.append(z)
                hs.append(h)
                recons.append(recon)
            zseq = torch.stack(zs, dim=1)
            hseq = torch.stack(hs, dim=1)
            recon = torch.stack(recons, dim=1)
            return recon, posts, priors, hseq, zseq

        @torch.no_grad()
        def posterior_latents(self, obs):
            recon, posts, priors, hseq, zseq = self.forward(obs)
            # Use posterior means as deterministic latent targets for CFM.
            means = torch.stack([p[0] for p in posts], dim=1)
            return hseq, means

    class CFMVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim = future_latent_dim + 1 + feat_dim
            self.net = nn.Sequential(
                nn.Linear(in_dim, args.hidden),
                nn.Tanh(),
                nn.Linear(args.hidden, args.hidden),
                nn.Tanh(),
                nn.Linear(args.hidden, future_latent_dim),
            )

        def forward(self, x):
            return self.net(x)

    def gaussian_kl(qm, qs, pm, ps):
        kl = torch.log(ps / qs) + (qs.pow(2) + (qm - pm).pow(2)) / (2.0 * ps.pow(2)) - 0.5
        kl = kl.sum(dim=-1)
        if args.free_nats > 0:
            kl = torch.clamp(kl, min=args.free_nats)
        return kl.mean()

    world = RSSMWorld().to(device)
    world_opt = torch.optim.AdamW(world.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train = torch.as_tensor(train_np, dtype=torch.float32, device=device)
    val = torch.as_tensor(val_np, dtype=torch.float32, device=device)

    metadata = {
        "method": "dreamer_posterior_latent_cfm",
        "args": vars(args),
        "dataset": asdict(stats),
        "obs_dim": obs_dim,
        "feat_dim": feat_dim,
        "future_latent_dim": future_latent_dim,
        "backend": "torch",
        "device": str(device),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    def batch_indices(n):
        return torch.randperm(n, device=device)

    @torch.no_grad()
    def eval_world():
        world.eval()
        losses = []
        for start in range(0, len(val), args.batch_size):
            obs = val[start : start + args.batch_size]
            recon, posts, priors, _, _ = world(obs)
            rec = F.mse_loss(recon, obs)
            kl = torch.stack([gaussian_kl(qm, qs, pm, ps) for (qm, qs), (pm, ps) in zip(posts, priors)]).mean()
            losses.append((rec + args.kl_scale * kl).item())
        world.train()
        return float(np.mean(losses)) if losses else 0.0

    best_world = float("inf")
    print(f"[World] device={device} train={len(train)} val={len(val)}")
    for ep in range(1, args.world_epochs + 1):
        perm = batch_indices(len(train))
        losses = []
        for start in range(0, len(train), args.batch_size):
            idx = perm[start : start + args.batch_size]
            obs = train[idx]
            recon, posts, priors, _, _ = world(obs)
            rec = F.mse_loss(recon, obs)
            kl = torch.stack([gaussian_kl(qm, qs, pm, ps) for (qm, qs), (pm, ps) in zip(posts, priors)]).mean()
            loss = rec + args.kl_scale * kl
            world_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(), args.grad_clip)
            world_opt.step()
            losses.append(loss.item())
        val_loss = eval_world()
        if val_loss < best_world:
            best_world = val_loss
            metadata["best_world_val"] = best_world
            metadata["best_world_epoch"] = ep
        print(f"[World {ep:04d}] train={float(np.mean(losses)):.6f} val={val_loss:.6f} best={best_world:.6f}")

    cfm = CFMVelocity().to(device)
    cfm_opt = torch.optim.AdamW(cfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def cfm_batch(obs):
        with torch.no_grad():
            hseq, zseq = world.posterior_latents(obs)
            cur_h = hseq[:, args.context_len - 1]
            cur_z = zseq[:, args.context_len - 1]
            cond = torch.cat([cur_h, cur_z], dim=-1)
            target = zseq[:, args.context_len : args.context_len + args.horizon].reshape(obs.shape[0], -1)
        z0 = torch.randn_like(target)
        tau = torch.rand((obs.shape[0], 1), device=device)
        xt = (1.0 - tau) * z0 + tau * target
        inp = torch.cat([xt, tau, cond], dim=-1)
        vel = target - z0
        return inp, vel

    @torch.no_grad()
    def eval_cfm():
        cfm.eval()
        losses = []
        for start in range(0, len(val), args.batch_size):
            obs = val[start : start + args.batch_size]
            inp, vel = cfm_batch(obs)
            pred = cfm(inp)
            losses.append(F.mse_loss(pred, vel).item())
        cfm.train()
        return float(np.mean(losses)) if losses else 0.0

    best_cfm = float("inf")
    best_path = str(outdir / "best_ckpt.pt")
    print(f"[CFM] target_dim={future_latent_dim}")
    for ep in range(1, args.cfm_epochs + 1):
        perm = batch_indices(len(train))
        losses = []
        for start in range(0, len(train), args.batch_size):
            obs = train[perm[start : start + args.batch_size]]
            inp, vel = cfm_batch(obs)
            pred = cfm(inp)
            loss = F.mse_loss(pred, vel)
            cfm_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), args.grad_clip)
            cfm_opt.step()
            losses.append(loss.item())
        val_loss = eval_cfm()
        if val_loss < best_cfm:
            best_cfm = val_loss
            metadata["best_cfm_val"] = best_cfm
            metadata["best_cfm_epoch"] = ep
            torch.save(
                {
                    "world_model": world.state_dict(),
                    "cfm_velocity": cfm.state_dict(),
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "metadata": metadata,
                },
                best_path,
            )
            with open(best_path + ".json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"[Save] {best_path}")
        elif ep % args.save_every == 0 or ep == args.cfm_epochs:
            path = str(outdir / f"ckpt_cfm_epoch_{ep:04d}.pt")
            torch.save(
                {
                    "world_model": world.state_dict(),
                    "cfm_velocity": cfm.state_dict(),
                    "obs_mean": obs_mean,
                    "obs_std": obs_std,
                    "metadata": metadata,
                },
                path,
            )
            print(f"[Save] {path}")
        print(f"[CFM {ep:04d}] train={float(np.mean(losses)):.6f} val={val_loss:.6f} best={best_cfm:.6f}")

    print(f"[Done] best_world={best_world:.6f} best_cfm={best_cfm:.6f}")
    print(f"[Done] best_ckpt={best_path}")


if __name__ == "__main__":
    main()
