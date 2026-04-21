#!/usr/bin/env python3
"""Formal NAVSIM offline pretraining: visual Dreamer posterior latents + CFM.

This is the formal stage-1 pretraining entrypoint. It uses NAVSIM image and ego
state sequences, trains a Dreamer-style posterior world model, then trains a
Conditional Flow Matching model over future posterior latents.

Checkpoint contents:
- visual/state RSSM world model
- CFM velocity network over posterior latent sequences
- observation normalization statistics
- full training metadata
"""

import argparse
import json
import math
import pathlib
import pickle
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


def yaw_from_matrix(ego2global: np.ndarray) -> float:
    rot = np.asarray(ego2global, dtype=np.float64)[:2, :2]
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def frame_state(frame: Dict) -> np.ndarray:
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
    yaw = yaw_from_matrix(frame["ego2global"])
    return np.concatenate(
        [dyn, cmd, np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32), can[10:18].astype(np.float32)],
        axis=0,
    ).astype(np.float32)


@dataclass
class DatasetStats:
    files_seen: int
    files_used: int
    files_bad: int
    sequences: int
    state_dim: int
    context_len: int
    horizon: int
    sample_every: int
    camera: str
    image_size: int


def load_metadata(data_dir, image_root, pattern, camera, context_len, horizon, sample_every, max_files, max_sequences):
    root = pathlib.Path(data_dir)
    image_root = pathlib.Path(image_root)
    files = sorted(root.rglob(pattern))
    if max_files > 0:
        files = files[:max_files]
    seq_len = context_len + horizon
    records = []
    states_all = []
    used = bad = 0
    for pkl_path in files:
        try:
            frames = pickle.load(open(pkl_path, "rb"))
            if not isinstance(frames, list) or len(frames) < seq_len:
                bad += 1
                continue
            states = np.stack([frame_state(f) for f in frames], axis=0).astype(np.float32)
            img_paths = []
            ok = True
            for f in frames:
                rel = f.get("cams", {}).get(camera, {}).get("data_path")
                if not rel:
                    ok = False
                    break
                path = image_root / rel
                if not path.exists():
                    ok = False
                    break
                img_paths.append(str(path))
            if not ok:
                bad += 1
                continue
            added = 0
            for start in range(0, len(frames) - seq_len + 1, sample_every):
                records.append((states[start : start + seq_len], img_paths[start : start + seq_len]))
                states_all.append(states[start : start + seq_len])
                added += 1
                if max_sequences > 0 and len(records) >= max_sequences:
                    break
            if added:
                used += 1
            if max_sequences > 0 and len(records) >= max_sequences:
                break
        except Exception:
            bad += 1
    if not records:
        raise RuntimeError("No valid NAVSIM image/state sequences found.")
    states_cat = np.concatenate(states_all, axis=0)
    mean = states_cat.mean(axis=0).astype(np.float32)
    std = np.maximum(states_cat.std(axis=0).astype(np.float32), 1e-6)
    return records, mean, std, used, bad, len(files)


class NavsimImageStateDataset:
    def __init__(self, records, state_mean, state_std, image_size):
        self.records = records
        self.state_mean = state_mean
        self.state_std = state_std
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        import torch

        states, paths = self.records[idx]
        states = ((states - self.state_mean) / self.state_std).astype(np.float32)
        imgs = []
        for path in paths:
            img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            imgs.append(arr)
        imgs = np.stack(imgs, axis=0)  # [T,H,W,3]
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0
        return torch.from_numpy(states), imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini")
    parser.add_argument("--image_root", default="/share/home/u23516/code/navsim_mini/mini_sensor_blobs/mini")
    parser.add_argument("--pattern", default="*.pkl")
    parser.add_argument("--camera", default="CAM_F0")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--outdir", default="dreamer/logs_navsim_formal_dreamer_cfm")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument("--context_len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--sample_every", type=int, default=2)
    parser.add_argument("--deter_dim", type=int, default=768)
    parser.add_argument("--stoch_dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--image_feat_dim", type=int, default=512)
    parser.add_argument("--world_epochs", type=int, default=300)
    parser.add_argument("--cfm_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--kl_scale", type=float, default=0.05)
    parser.add_argument("--image_recon_scale", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()
    if args.image_size != 96:
        raise ValueError("The formal image decoder currently expects --image_size 96.")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    run_name = args.run_name or time.strftime("navsim_formal_dreamer_cfm_%Y%m%d_%H%M%S")
    outdir = pathlib.Path(args.outdir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Load] data={args.data_dir} image_root={args.image_root} camera={args.camera}")
    records, state_mean, state_std, used, bad, seen = load_metadata(
        args.data_dir, args.image_root, args.pattern, args.camera,
        args.context_len, args.horizon, args.sample_every, args.max_files, args.max_sequences)
    seq_len = args.context_len + args.horizon
    state_dim = records[0][0].shape[-1]
    stats = DatasetStats(seen, used, bad, len(records), state_dim, args.context_len, args.horizon, args.sample_every, args.camera, args.image_size)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    cut = max(1, int(len(idx) * (1 - args.val_ratio)))
    ds = NavsimImageStateDataset(records, state_mean, state_std, args.image_size)
    train_loader = DataLoader(Subset(ds, idx[:cut].tolist()), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, idx[cut:].tolist()), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print(f"[Data] {asdict(stats)} train={cut} val={len(idx)-cut}")

    feat_dim = args.deter_dim + args.stoch_dim
    future_latent_dim = args.horizon * args.stoch_dim

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ELU(),
                nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ELU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ELU(),
                nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.proj = nn.Linear(256, args.image_feat_dim)
        def forward(self, x):
            return self.proj(self.net(x).flatten(1))

    class ImageDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(nn.Linear(feat_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 256 * 6 * 6), nn.ELU())
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ELU(),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ELU(),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ELU(),
                nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
            )
        def forward(self, feat):
            x = self.fc(feat).view(feat.shape[0], 256, 6, 6)
            return self.deconv(x)

    class RSSMWorld(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_enc = ImageEncoder()
            self.state_enc = nn.Sequential(nn.Linear(state_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 256), nn.ELU())
            token_dim = args.image_feat_dim + 256
            self.gru = nn.GRUCell(args.stoch_dim, args.deter_dim)
            self.post = nn.Linear(args.deter_dim + token_dim, 2 * args.stoch_dim)
            self.prior = nn.Sequential(nn.Linear(args.deter_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, 2 * args.stoch_dim))
            self.state_dec = nn.Sequential(nn.Linear(feat_dim, args.hidden), nn.ELU(), nn.Linear(args.hidden, args.hidden), nn.ELU(), nn.Linear(args.hidden, state_dim))
            self.img_dec = ImageDecoder()
        def dist(self, x):
            mean, rawstd = x.chunk(2, -1)
            return mean, F.softplus(rawstd) + 0.1
        def token(self, state, image):
            b, t = state.shape[:2]
            img = self.img_enc(image.reshape(b * t, 3, args.image_size, args.image_size)).reshape(b, t, -1)
            st = self.state_enc(state.reshape(b * t, state_dim)).reshape(b, t, -1)
            return torch.cat([img, st], -1)
        def forward(self, state, image):
            b, t = state.shape[:2]
            tokens = self.token(state, image)
            h = torch.zeros((b, args.deter_dim), device=state.device)
            z = torch.zeros((b, args.stoch_dim), device=state.device)
            posts, priors, zs, hs, state_recons, img_recons = [], [], [], [], [], []
            for i in range(t):
                h = self.gru(z, h)
                pm, ps = self.dist(self.prior(h))
                qm, qs = self.dist(self.post(torch.cat([h, tokens[:, i]], -1)))
                z = qm + torch.randn_like(qm) * qs
                feat = torch.cat([h, z], -1)
                posts.append((qm, qs)); priors.append((pm, ps)); zs.append(z); hs.append(h)
                state_recons.append(self.state_dec(feat))
                img_recons.append(self.img_dec(feat))
            return (
                torch.stack(state_recons, 1),
                torch.stack(img_recons, 1),
                posts,
                priors,
                torch.stack(hs, 1),
                torch.stack(zs, 1),
            )
        @torch.no_grad()
        def posterior_latents(self, state, image):
            _, _, posts, _, hseq, _ = self.forward(state, image)
            return hseq, torch.stack([p[0] for p in posts], 1)

    class CFMVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(future_latent_dim + 1 + feat_dim, args.hidden), nn.Tanh(),
                nn.Linear(args.hidden, args.hidden), nn.Tanh(),
                nn.Linear(args.hidden, future_latent_dim),
            )
        def forward(self, x):
            return self.net(x)

    def kl(qm, qs, pm, ps):
        out = torch.log(ps / qs) + (qs.square() + (qm - pm).square()) / (2 * ps.square()) - 0.5
        return out.sum(-1).mean()

    world = RSSMWorld().to(device)
    opt = torch.optim.AdamW(world.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metadata = {"method": "formal_visual_dreamer_posterior_latent_cfm", "args": vars(args), "dataset": asdict(stats), "feat_dim": feat_dim, "future_latent_dim": future_latent_dim, "device": str(device)}
    (outdir / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def move(batch):
        state, image = batch
        return state.to(device, non_blocking=True), image.to(device, non_blocking=True)

    @torch.no_grad()
    def eval_world():
        world.eval(); vals = []
        for batch in val_loader:
            state, image = move(batch)
            sr, ir, posts, priors, _, _ = world(state, image)
            rec_s = F.mse_loss(sr, state)
            rec_i = F.mse_loss(ir, image)
            k = torch.stack([kl(qm, qs, pm, ps) for (qm, qs), (pm, ps) in zip(posts, priors)]).mean()
            vals.append((rec_s + args.image_recon_scale * rec_i + args.kl_scale * k).item())
        world.train(); return float(np.mean(vals))

    best_world = float("inf")
    best_world_path = str(outdir / "best_world_ckpt.pt")
    print(f"[World] device={device} train={cut} val={len(idx)-cut}")
    for ep in range(1, args.world_epochs + 1):
        losses = []
        for batch in train_loader:
            state, image = move(batch)
            sr, ir, posts, priors, _, _ = world(state, image)
            rec_s = F.mse_loss(sr, state)
            rec_i = F.mse_loss(ir, image)
            k = torch.stack([kl(qm, qs, pm, ps) for (qm, qs), (pm, ps) in zip(posts, priors)]).mean()
            loss = rec_s + args.image_recon_scale * rec_i + args.kl_scale * k
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(world.parameters(), args.grad_clip); opt.step()
            losses.append(loss.item())
        val = eval_world()
        if val < best_world:
            best_world = val
            metadata["best_world_val"] = best_world
            metadata["best_world_epoch"] = ep
            torch.save({"world_model": world.state_dict(), "state_mean": state_mean, "state_std": state_std, "metadata": metadata}, best_world_path)
            print(f"[Save] {best_world_path}")
        print(f"[World {ep:04d}] train={float(np.mean(losses)):.6f} val={val:.6f} best={best_world:.6f}")

    cfm = CFMVelocity().to(device)
    cfm_opt = torch.optim.AdamW(cfm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def cfm_batch(state, image):
        with torch.no_grad():
            hseq, zseq = world.posterior_latents(state, image)
            cond = torch.cat([hseq[:, args.context_len - 1], zseq[:, args.context_len - 1]], -1)
            target = zseq[:, args.context_len:args.context_len + args.horizon].reshape(state.shape[0], -1)
        z0 = torch.randn_like(target); tau = torch.rand((target.shape[0], 1), device=device)
        xt = (1 - tau) * z0 + tau * target
        return torch.cat([xt, tau, cond], -1), target - z0

    @torch.no_grad()
    def eval_cfm():
        cfm.eval(); vals = []
        for batch in val_loader:
            state, image = move(batch); inp, vel = cfm_batch(state, image)
            vals.append(F.mse_loss(cfm(inp), vel).item())
        cfm.train(); return float(np.mean(vals))

    best_cfm = float("inf"); best_path = str(outdir / "best_ckpt.pt")
    print(f"[CFM] target_dim={future_latent_dim}")
    for ep in range(1, args.cfm_epochs + 1):
        losses = []
        for batch in train_loader:
            state, image = move(batch); inp, vel = cfm_batch(state, image)
            loss = F.mse_loss(cfm(inp), vel)
            cfm_opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(cfm.parameters(), args.grad_clip); cfm_opt.step()
            losses.append(loss.item())
        val = eval_cfm()
        if val < best_cfm:
            best_cfm = val
            metadata["best_cfm_val"] = best_cfm; metadata["best_cfm_epoch"] = ep
            torch.save({"world_model": world.state_dict(), "cfm_velocity": cfm.state_dict(), "state_mean": state_mean, "state_std": state_std, "metadata": metadata}, best_path)
            (pathlib.Path(best_path + ".json")).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[Save] {best_path}")
        elif ep % args.save_every == 0 or ep == args.cfm_epochs:
            path = str(outdir / f"ckpt_cfm_epoch_{ep:04d}.pt")
            torch.save({"world_model": world.state_dict(), "cfm_velocity": cfm.state_dict(), "state_mean": state_mean, "state_std": state_std, "metadata": metadata}, path)
            print(f"[Save] {path}")
        print(f"[CFM {ep:04d}] train={float(np.mean(losses)):.6f} val={val:.6f} best={best_cfm:.6f}")
    print(f"[Done] best_world={best_world:.6f} best_cfm={best_cfm:.6f}")
    print(f"[Done] best_ckpt={best_path}")


if __name__ == "__main__":
    main()
