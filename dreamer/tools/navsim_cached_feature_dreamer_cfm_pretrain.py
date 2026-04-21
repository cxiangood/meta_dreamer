#!/usr/bin/env python3
"""NAVSIM formal offline pretraining with cached visual features.

Stage 1 builds a reusable cache from NAVSIM camera frames:
- unique image path table
- deterministic multi-scale visual features for each unique frame
- normalized ego/state sequences and frame-index sequences

Stage 2 trains a Dreamer-style RSSM posterior latent world model from cached
visual features + states, then trains a CFM velocity model over future posterior
latents. This avoids repeated JPG decoding and NFS reads in every epoch.
"""

import argparse
import json
import math
import pathlib
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


def yaw_from_matrix(ego2global: np.ndarray) -> float:
    rot = np.asarray(ego2global, dtype=np.float64)[:2, :2]
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def frame_state(frame: Dict) -> np.ndarray:
    dyn = np.asarray(frame.get("ego_dynamic_state", [0.0, 0.0, 0.0, 0.0]), dtype=np.float32)
    if dyn.shape[0] < 4:
        dyn = np.pad(dyn, (0, 4 - dyn.shape[0]))
    cmd = np.asarray(frame.get("driving_command", np.zeros(4)), dtype=np.float32)
    if cmd.shape[0] < 4:
        cmd = np.pad(cmd, (0, 4 - cmd.shape[0]))
    can = np.asarray(frame.get("can_bus", np.zeros(18)), dtype=np.float32)
    if can.shape[0] < 18:
        can = np.pad(can, (0, 18 - can.shape[0]))
    yaw = yaw_from_matrix(frame["ego2global"])
    return np.concatenate(
        [dyn[:4], cmd[:4], np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32), can[10:18]],
        axis=0,
    ).astype(np.float32)


def visual_feature_dim(grid_size: int) -> int:
    return 4 * grid_size * grid_size + 6


def pooled_visual_features(path: str, image_size: int, grid_size: int) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    cell = image_size // grid_size
    rgb = arr.reshape(grid_size, cell, grid_size, cell, 3).mean(axis=(1, 3))
    gray = arr.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    grad = np.sqrt(gx * gx + gy * gy)
    grad_grid = grad.reshape(grid_size, cell, grid_size, cell).mean(axis=(1, 3))
    global_stats = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))]).astype(np.float32)
    return np.concatenate([rgb.reshape(-1), grad_grid.reshape(-1), global_stats], axis=0).astype(np.float32)


@dataclass
class CacheStats:
    files_seen: int
    files_used: int
    files_bad: int
    sequences: int
    unique_frames: int
    state_dim: int
    visual_dim: int
    context_len: int
    horizon: int
    sample_every: int
    camera: str
    image_size: int
    grid_size: int


def collect_navsim_sequences(args):
    root = pathlib.Path(args.data_dir)
    image_root = pathlib.Path(args.image_root)
    files = sorted(root.rglob(args.pattern))
    if args.max_files > 0:
        files = files[: args.max_files]
    seq_len = args.context_len + args.horizon
    seq_states = []
    seq_paths = []
    frame_to_idx = {}
    frame_paths = []
    used = bad = 0
    for pkl_path in files:
        try:
            frames = pickle.load(open(pkl_path, "rb"))
            if not isinstance(frames, list) or len(frames) < seq_len:
                bad += 1
                continue
            states = np.stack([frame_state(f) for f in frames], axis=0).astype(np.float32)
            paths = []
            ok = True
            for frame in frames:
                rel = frame.get("cams", {}).get(args.camera, {}).get("data_path")
                if not rel:
                    ok = False
                    break
                img_path = str(image_root / rel)
                if not pathlib.Path(img_path).exists():
                    ok = False
                    break
                if img_path not in frame_to_idx:
                    frame_to_idx[img_path] = len(frame_paths)
                    frame_paths.append(img_path)
                paths.append(frame_to_idx[img_path])
            if not ok:
                bad += 1
                continue
            added = 0
            for start in range(0, len(frames) - seq_len + 1, args.sample_every):
                seq_states.append(states[start : start + seq_len])
                seq_paths.append(paths[start : start + seq_len])
                added += 1
                if args.max_sequences > 0 and len(seq_states) >= args.max_sequences:
                    break
            if added:
                used += 1
            if args.max_sequences > 0 and len(seq_states) >= args.max_sequences:
                break
        except Exception as exc:
            print(f"[Cache] skip {pkl_path}: {exc}", flush=True)
            bad += 1
    if not seq_states:
        raise RuntimeError("No valid NAVSIM sequences found.")
    return files, used, bad, np.stack(seq_states), np.asarray(seq_paths, dtype=np.int32), frame_paths


def cache_meta_matches(meta_path: pathlib.Path, args) -> bool:
    if args.rebuild_cache or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "data_dir": args.data_dir,
        "image_root": args.image_root,
        "pattern": args.pattern,
        "camera": args.camera,
        "image_size": args.image_size,
        "grid_size": args.grid_size,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "sample_every": args.sample_every,
        "max_files": args.max_files,
        "max_sequences": args.max_sequences,
    }
    return all(meta.get(k) == v for k, v in expected.items())


def build_or_load_cache(args):
    cache_dir = pathlib.Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cache_meta.json"
    seq_states_path = cache_dir / "seq_states.npy"
    seq_frame_idx_path = cache_dir / "seq_frame_indices.npy"
    frame_features_path = cache_dir / "frame_features.npy"
    state_mean_path = cache_dir / "state_mean.npy"
    state_std_path = cache_dir / "state_std.npy"
    frame_paths_path = cache_dir / "frame_paths.json"

    if (
        cache_meta_matches(meta_path, args)
        and seq_states_path.exists()
        and seq_frame_idx_path.exists()
        and frame_features_path.exists()
        and state_mean_path.exists()
        and state_std_path.exists()
    ):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[Cache] reuse {cache_dir}", flush=True)
        return meta

    print(f"[Cache] build {cache_dir}", flush=True)
    files, used, bad, seq_states, seq_frame_idx, frame_paths = collect_navsim_sequences(args)
    state_mean = seq_states.reshape(-1, seq_states.shape[-1]).mean(axis=0).astype(np.float32)
    state_std = np.maximum(seq_states.reshape(-1, seq_states.shape[-1]).std(axis=0).astype(np.float32), 1e-6)
    norm_states = ((seq_states - state_mean) / state_std).astype(np.float32)

    np.save(seq_states_path, norm_states)
    np.save(seq_frame_idx_path, seq_frame_idx)
    np.save(state_mean_path, state_mean)
    np.save(state_std_path, state_std)
    frame_paths_path.write_text(json.dumps(frame_paths, ensure_ascii=False), encoding="utf-8")

    vdim = visual_feature_dim(args.grid_size)
    features = np.lib.format.open_memmap(
        frame_features_path, mode="w+", dtype=np.float16, shape=(len(frame_paths), vdim)
    )
    start = time.time()
    for idx, path in enumerate(frame_paths):
        features[idx] = pooled_visual_features(path, args.image_size, args.grid_size).astype(np.float16)
        if (idx + 1) % args.cache_log_every == 0 or idx + 1 == len(frame_paths):
            elapsed = max(time.time() - start, 1e-6)
            rate = (idx + 1) / elapsed
            remain = (len(frame_paths) - idx - 1) / max(rate, 1e-6)
            print(
                f"[Cache] frame {idx + 1}/{len(frame_paths)} rate={rate:.1f}/s eta={remain/60:.1f}m",
                flush=True,
            )
    del features

    stats = CacheStats(
        files_seen=len(files),
        files_used=used,
        files_bad=bad,
        sequences=int(seq_states.shape[0]),
        unique_frames=len(frame_paths),
        state_dim=int(seq_states.shape[-1]),
        visual_dim=vdim,
        context_len=args.context_len,
        horizon=args.horizon,
        sample_every=args.sample_every,
        camera=args.camera,
        image_size=args.image_size,
        grid_size=args.grid_size,
    )
    meta = {
        "data_dir": args.data_dir,
        "image_root": args.image_root,
        "pattern": args.pattern,
        "camera": args.camera,
        "image_size": args.image_size,
        "grid_size": args.grid_size,
        "context_len": args.context_len,
        "horizon": args.horizon,
        "sample_every": args.sample_every,
        "max_files": args.max_files,
        "max_sequences": args.max_sequences,
        "stats": asdict(stats),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Cache] ready {asdict(stats)}", flush=True)
    return meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini")
    parser.add_argument("--image_root", default="/share/home/u23516/code/navsim_mini/mini_sensor_blobs/mini")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--camera", default="CAM_F0")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--grid_size", type=int, default=12)
    parser.add_argument("--cache_dir", default="/share/home/u23516/code/meta_dreamer-sub/dreamer/navsim_feature_cache/formal_cam_f0_s96_g12")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--cache_log_every", type=int, default=1000)
    parser.add_argument("--outdir", default="dreamer/logs_navsim_cached_feature_dreamer_cfm")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--context_len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--deter_dim", type=int, default=512)
    parser.add_argument("--stoch_dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--world_epochs", type=int, default=120)
    parser.add_argument("--cfm_epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--visual_recon_scale", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every_batches", type=int, default=25)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_size % args.grid_size != 0:
        raise ValueError("--image_size must be divisible by --grid_size")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, Subset

    meta = build_or_load_cache(args)
    stats = meta["stats"]
    run_name = args.run_name or time.strftime("navsim_cached_feature_dreamer_cfm_%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    seq_states = np.load(pathlib.Path(args.cache_dir) / "seq_states.npy", mmap_mode="r")
    seq_frame_idx = np.load(pathlib.Path(args.cache_dir) / "seq_frame_indices.npy", mmap_mode="r")
    frame_features = np.load(pathlib.Path(args.cache_dir) / "frame_features.npy", mmap_mode="r")
    state_mean = np.load(pathlib.Path(args.cache_dir) / "state_mean.npy")
    state_std = np.load(pathlib.Path(args.cache_dir) / "state_std.npy")

    class CachedDataset(Dataset):
        def __len__(self):
            return int(seq_states.shape[0])
        def __getitem__(self, index):
            state = np.array(seq_states[index], dtype=np.float32, copy=True)
            vision = np.array(frame_features[np.asarray(seq_frame_idx[index])], dtype=np.float32, copy=True)
            return torch.from_numpy(state), torch.from_numpy(vision)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    idx = np.arange(len(seq_states))
    np.random.default_rng(args.seed).shuffle(idx)
    cut = max(1, int(len(idx) * (1 - args.val_ratio)))
    ds = CachedDataset()
    train_loader = DataLoader(Subset(ds, idx[:cut].tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, idx[cut:].tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    state_dim = int(seq_states.shape[-1])
    visual_dim = int(frame_features.shape[-1])
    feat_dim = args.deter_dim + args.stoch_dim
    future_latent_dim = args.horizon * args.stoch_dim
    metadata = {
        "method": "cached_visual_feature_dreamer_posterior_latent_cfm",
        "args": vars(args),
        "cache": stats,
        "state_dim": state_dim,
        "visual_dim": visual_dim,
        "feat_dim": feat_dim,
        "future_latent_dim": future_latent_dim,
        "device": str(device),
    }
    (outdir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Data] cache={args.cache_dir} stats={stats} train={cut} val={len(idx)-cut}", flush=True)

    class RSSMWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_enc = nn.Sequential(nn.Linear(state_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 256), nn.ELU())
            self.visual_enc = nn.Sequential(nn.Linear(visual_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 256), nn.ELU())
            token_dim = 512
            self.gru = nn.GRUCell(args.stoch_dim, args.deter_dim)
            self.post = nn.Linear(args.deter_dim + token_dim, 2 * args.stoch_dim)
            self.prior = nn.Sequential(nn.Linear(args.deter_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 2 * args.stoch_dim))
            self.state_dec = nn.Sequential(nn.Linear(feat_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, state_dim))
            self.visual_dec = nn.Sequential(nn.Linear(feat_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, visual_dim))
        def dist(self, x):
            mean, rawstd = x.chunk(2, -1)
            return mean, F.softplus(rawstd) + 0.1
        def token(self, state, vision):
            b, t = state.shape[:2]
            st = self.state_enc(state.reshape(b * t, state_dim)).reshape(b, t, -1)
            vi = self.visual_enc(vision.reshape(b * t, visual_dim)).reshape(b, t, -1)
            return torch.cat([st, vi], -1)
        def forward(self, state, vision):
            b, t = state.shape[:2]
            tokens = self.token(state, vision)
            h = torch.zeros((b, args.deter_dim), device=state.device)
            z = torch.zeros((b, args.stoch_dim), device=state.device)
            posts, priors, hs, zs, state_recons, visual_recons = [], [], [], [], [], []
            for i in range(t):
                h = self.gru(z, h)
                pm, ps = self.dist(self.prior(h))
                qm, qs = self.dist(self.post(torch.cat([h, tokens[:, i]], -1)))
                z = qm + torch.randn_like(qm) * qs
                feat = torch.cat([h, z], -1)
                posts.append((qm, qs)); priors.append((pm, ps)); hs.append(h); zs.append(z)
                state_recons.append(self.state_dec(feat))
                visual_recons.append(self.visual_dec(feat))
            return torch.stack(state_recons, 1), torch.stack(visual_recons, 1), posts, priors, torch.stack(hs, 1), torch.stack(zs, 1)
        @torch.no_grad()
        def posterior_latents(self, state, vision):
            _, _, posts, _, hseq, _ = self.forward(state, vision)
            return hseq, torch.stack([p[0] for p in posts], 1)

    class CFMVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(future_latent_dim + 1 + feat_dim, args.hidden), nn.SiLU(),
                nn.Linear(args.hidden, args.hidden), nn.SiLU(),
                nn.Linear(args.hidden, future_latent_dim),
            )
        def forward(self, x):
            return self.net(x)

    def kl(qm, qs, pm, ps):
        return (torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - 0.5).sum(-1).mean()

    def move(batch):
        state, vision = batch
        return state.to(device, non_blocking=True), vision.to(device, non_blocking=True)

    world = RSSMWorld().to(device)
    opt = torch.optim.AdamW(world.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    @torch.no_grad()
    def eval_world():
        world.eval(); vals = []
        for batch in val_loader:
            state, vision = move(batch)
            sr, vr, posts, priors, _, _ = world(state, vision)
            rec_s = F.mse_loss(sr, state)
            rec_v = F.mse_loss(vr, vision)
            k = torch.stack([kl(qm, qs, pm, ps) for (qm, qs), (pm, ps) in zip(posts, priors)]).mean()
            vals.append((rec_s + args.visual_recon_scale * rec_v + args.kl_scale * k).item())
        world.train()
        return float(np.mean(vals))

    best_world = float("inf")
    best_world_path = str(outdir / "best_world_ckpt.pt")
    print(f"[World] device={device} batches={len(train_loader)} val_batches={len(val_loader)}", flush=True)
    for ep in range(1, args.world_epochs + 1):
        losses = []
        start = time.time()
        for bi, batch in enumerate(train_loader, start=1):
            state, vision = move(batch)
            sr, vr, posts, priors, _, _ = world(state, vision)
            rec_s = F.mse_loss(sr, state)
            rec_v = F.mse_loss(vr, vision)
            k = torch.stack([kl(qm, qs, pm, ps) for (qm, qs), (pm, ps) in zip(posts, priors)]).mean()
            loss = rec_s + args.visual_recon_scale * rec_v + args.kl_scale * k
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world.parameters(), args.grad_clip)
            opt.step()
            losses.append(loss.item())
            if args.log_every_batches > 0 and bi % args.log_every_batches == 0:
                elapsed = max(time.time() - start, 1e-6)
                eta = (len(train_loader) - bi) * elapsed / bi / 60.0
                print(f"[World {ep:04d}] batch={bi}/{len(train_loader)} loss={loss.item():.6f} eta_epoch={eta:.1f}m", flush=True)
        val = eval_world()
        if val < best_world:
            best_world = val
            metadata["best_world_val"] = best_world
            metadata["best_world_epoch"] = ep
            torch.save({"world_model": world.state_dict(), "state_mean": state_mean, "state_std": state_std, "metadata": metadata}, best_world_path)
            print(f"[Save] {best_world_path}", flush=True)
        print(f"[World {ep:04d}] train={float(np.mean(losses)):.6f} val={val:.6f} best={best_world:.6f} epoch_time={(time.time()-start)/60:.1f}m", flush=True)

    cfm = CFMVelocity().to(device)
    cfm_opt = torch.optim.AdamW(cfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def cfm_batch(state, vision):
        with torch.no_grad():
            hseq, zseq = world.posterior_latents(state, vision)
            cond = torch.cat([hseq[:, args.context_len - 1], zseq[:, args.context_len - 1]], -1)
            target = zseq[:, args.context_len : args.context_len + args.horizon].reshape(state.shape[0], -1)
        z0 = torch.randn_like(target)
        tau = torch.rand((target.shape[0], 1), device=device)
        xt = (1 - tau) * z0 + tau * target
        return torch.cat([xt, tau, cond], -1), target - z0

    @torch.no_grad()
    def eval_cfm():
        cfm.eval(); vals = []
        for batch in val_loader:
            state, vision = move(batch)
            inp, vel = cfm_batch(state, vision)
            vals.append(F.mse_loss(cfm(inp), vel).item())
        cfm.train()
        return float(np.mean(vals))

    best_cfm = float("inf")
    best_path = str(outdir / "best_ckpt.pt")
    print(f"[CFM] target_dim={future_latent_dim} batches={len(train_loader)}", flush=True)
    for ep in range(1, args.cfm_epochs + 1):
        losses = []
        start = time.time()
        for bi, batch in enumerate(train_loader, start=1):
            state, vision = move(batch)
            inp, vel = cfm_batch(state, vision)
            loss = F.mse_loss(cfm(inp), vel)
            cfm_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cfm.parameters(), args.grad_clip)
            cfm_opt.step()
            losses.append(loss.item())
            if args.log_every_batches > 0 and bi % args.log_every_batches == 0:
                elapsed = max(time.time() - start, 1e-6)
                eta = (len(train_loader) - bi) * elapsed / bi / 60.0
                print(f"[CFM {ep:04d}] batch={bi}/{len(train_loader)} loss={loss.item():.6f} eta_epoch={eta:.1f}m", flush=True)
        val = eval_cfm()
        if val < best_cfm:
            best_cfm = val
            metadata["best_cfm_val"] = best_cfm
            metadata["best_cfm_epoch"] = ep
            torch.save({"world_model": world.state_dict(), "cfm_velocity": cfm.state_dict(), "state_mean": state_mean, "state_std": state_std, "metadata": metadata}, best_path)
            pathlib.Path(best_path + ".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Save] {best_path}", flush=True)
        elif ep % args.save_every == 0 or ep == args.cfm_epochs:
            path = str(outdir / f"ckpt_cfm_epoch_{ep:04d}.pt")
            torch.save({"world_model": world.state_dict(), "cfm_velocity": cfm.state_dict(), "state_mean": state_mean, "state_std": state_std, "metadata": metadata}, path)
            print(f"[Save] {path}", flush=True)
        print(f"[CFM {ep:04d}] train={float(np.mean(losses)):.6f} val={val:.6f} best={best_cfm:.6f} epoch_time={(time.time()-start)/60:.1f}m", flush=True)
    print(f"[Done] best_world={best_world:.6f} best_cfm={best_cfm:.6f}", flush=True)
    print(f"[Done] best_ckpt={best_path}", flush=True)


if __name__ == "__main__":
    main()
