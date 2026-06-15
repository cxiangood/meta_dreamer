"""
Offline Dataset: Loads pre-collected exiD .npz files for world model training.

Each .npz contains:
  bev_images:   (T, H, W, 3) uint8  — BEV RGB images
  actions:      (T, 2) float32       — [steering, throttle]
  rewards:      (T,)  float32        — heuristic reward
  dones:        (T,)  bool           — episode termination
  positions:    (T, 2) float32       — [x, y] (optional)
  lanelet_ids:  (T,)  int32          — lanelet IDs (optional)

Usage:
  loader = OfflineDataset(data_dir, bev_size=64, seq_len=50, cache_size=16)
  obs, actions, rewards, continues = loader.sample(batch_size)
"""

import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque


class OfflineDataset:
    def __init__(self, data_dir, bev_size=64, seq_len=50,
                 cache_size=32, device="cpu", preload=False,
                 skip_resize=False, return_phase_labels=False,
                 merge_zone_frames=20, return_speed_labels=False,
                 return_positions=False, traj_horizon=10):
        self.data_dir = data_dir
        self.bev_size = bev_size
        self.seq_len = seq_len
        self.device = device
        self.cache_size = cache_size
        self.preload = preload
        self.skip_resize = skip_resize
        self.return_phase_labels = return_phase_labels
        self.merge_zone_frames = merge_zone_frames
        self.return_speed_labels = return_speed_labels
        self.return_positions = return_positions
        self.traj_horizon = traj_horizon

        # Scan all npz files
        self._file_list = sorted(glob.glob(
            os.path.join(data_dir, "**", "track*.npz"), recursive=True
        ))
        if not self._file_list:
            raise FileNotFoundError(f"No track*.npz found under {data_dir}")

        # File length index
        self._file_lengths = {}

        print(f"[OfflineDataset] {len(self._file_list)} files found")
        print(f"[OfflineDataset] File sample: {self._file_list[0]}")

        if preload:
            self._preload_all()
        else:
            # File cache (LRU via deque)
            self._cache = {}
            self._cache_q = deque(maxlen=cache_size)
            self._preloaded = None

    def _preload_all(self):
        """Load all npz files into RAM. ~3GB for 6329 files, fits in 64GB HPC RAM."""
        import time
        t0 = time.time()
        self._preloaded = []
        skipped = 0
        for fpath in self._file_list:
            try:
                data = dict(np.load(fpath))
                T = data["actions"].shape[0]
                if T < self.seq_len + 1:
                    skipped += 1
                    continue
                # Pre-process BEV images: resize to target, store as uint8 for RAM
                bev = torch.from_numpy(data["bev_images"].copy()).permute(0, 3, 1, 2).float()
                H, W = bev.shape[-2], bev.shape[-1]
                if H != W:
                    size = min(H, W)
                    dh = (H - size) // 2
                    dw = (W - size) // 2
                    bev = bev[:, :, dh:dh + size, dw:dw + size]
                if size != self.bev_size:
                    bev = F.interpolate(bev, size=(self.bev_size, self.bev_size),
                                        mode='bilinear', align_corners=False)
                bev = bev.round().clamp(0, 255).to(torch.uint8)
                entry = {
                    "bev": bev,                    # (T, 3, H, W) uint8
                    "actions": data["actions"],    # (T, 2) float32
                    "rewards": data["rewards"],    # (T,) float32
                    "dones": data["dones"],        # (T,) bool
                }
                if self.return_phase_labels:
                    entry["merge_frame_idx"] = int(data.get("merge_frame_idx", -1))
                if self.return_speed_labels or self.return_positions:
                    positions = data.get("positions", None)
                    if positions is not None and len(positions) > 1:
                        if self.return_positions:
                            entry["positions"] = positions.astype(np.float32)
                        if self.return_speed_labels:
                            # Speed from position deltas at 25fps (ΔT=0.04s)
                            deltas = np.diff(positions, axis=0)
                            speeds = np.linalg.norm(deltas, axis=1) / 0.04
                            speeds = np.concatenate([[speeds[0]], speeds])
                            entry["speeds"] = speeds.astype(np.float32)
                    elif self.return_speed_labels:
                        entry["speeds"] = np.zeros(entry["actions"].shape[0], dtype=np.float32)
                self._preloaded.append(entry)
            except Exception:
                skipped += 1
        self._preload_lengths = [d["actions"].shape[0] for d in self._preloaded]
        total_frames = sum(self._preload_lengths)
        mem_mb = sum(d["bev"].numel() for d in self._preloaded) / 1e6  # uint8 = 1 byte/element
        print(f"[OfflineDataset] Preloaded {len(self._preloaded)} files "
              f"({total_frames} frames, ~{mem_mb:.0f}MB) in {time.time()-t0:.1f}s "
              f"(skipped {skipped})")

    def __len__(self):
        if self.preload:
            return len(self._preloaded)
        return len(self._file_list)

    def _get_file_len(self, fidx):
        """Lazily get the number of frames in a file."""
        if fidx not in self._file_lengths:
            try:
                with np.load(self._file_list[fidx], mmap_mode='r') as d:
                    self._file_lengths[fidx] = d["actions"].shape[0]
            except Exception:
                self._file_lengths[fidx] = 0
        return self._file_lengths[fidx]

    def _load_file(self, fidx):
        """Load npz file with LRU cache (mmap mode for lazy IO)."""
        if fidx in self._cache:
            return self._cache[fidx]
        # mmap_mode='r' defers reads until specific arrays/indices are accessed
        data = np.load(self._file_list[fidx], mmap_mode='r')
        if len(self._cache_q) >= self.cache_size:
            old = self._cache_q.popleft()
            # Close mmap before evicting
            if hasattr(self._cache[old], 'close'):
                try:
                    self._cache[old].close()
                except Exception:
                    pass
            del self._cache[old]
        self._cache[fidx] = data
        self._cache_q.append(fidx)
        return data

    def sample(self, batch_size):
        """Sample random sequences. Uses preloaded data if available (fast)."""
        if self.preload:
            return self._sample_preloaded(batch_size)
        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_con = []
        batch_phase = []
        batch_speed = []
        batch_positions = []

        # Need seq_len+H+1 frames for trajectory labels (H future frames per feature)
        min_frames = self.seq_len + 1
        if self.return_positions:
            min_frames = max(min_frames, self.seq_len + self.traj_horizon + 1)

        for _ in range(batch_size):
            # Pick a random file with valid length
            for _ in range(50):  # max retries
                fidx = np.random.randint(0, len(self._file_list))
                T = self._get_file_len(fidx)
                if T >= min_frames:
                    break
            else:
                raise RuntimeError("No valid files found after 50 retries")

            # Random start frame
            start = np.random.randint(0, T - self.seq_len - self.traj_horizon)
            end = start + self.seq_len
            data = self._load_file(fidx)

            # Extract sequence: need seq_len+1 frames (obs at t and t+1)
            bev_seq = data["bev_images"][start:end + 1]      # (seq_len+1, H, W, 3)
            act_seq = data["actions"][start:end]              # (seq_len, 2)
            rew_seq = data["rewards"][start:end]              # (seq_len,)
            don_seq = data["dones"][start:end]                # (seq_len,)

            # Resize BEV images: (T, H, W, 3) -> (T, 3, H, W), normalize [0,255] -> [0,1]
            bev_tensor = torch.from_numpy(bev_seq).permute(0, 3, 1, 2).float() / 255.0

            # Center-crop to square to preserve aspect ratio
            H, W = bev_tensor.shape[-2], bev_tensor.shape[-1]
            if H != W:
                size = min(H, W)
                dh = (H - size) // 2
                dw = (W - size) // 2
                bev_tensor = bev_tensor[:, :, dh:dh + size, dw:dw + size]

            # Resize to target (unless using CNN frontend, which handles downsampling)
            if not self.skip_resize and size != self.bev_size:
                bev_tensor = F.interpolate(
                    bev_tensor, size=(self.bev_size, self.bev_size),
                    mode='bilinear', align_corners=False
                )

            batch_obs.append(bev_tensor)
            batch_act.append(torch.from_numpy(act_seq))
            batch_rew.append(torch.from_numpy(rew_seq))
            batch_con.append(torch.from_numpy(~don_seq).float())

            if self.return_phase_labels:
                mf = int(data.get("merge_frame_idx", -1) if "merge_frame_idx" in data else -1)
                frames = np.arange(start, end)
                phase = np.zeros(self.seq_len, dtype=np.int64)
                if mf >= 0:
                    phase[frames >= mf] = 1                         # merge-zone
                    phase[frames >= mf + self.merge_zone_frames] = 2  # post-merge (applied after, not overwritten)
                batch_phase.append(torch.from_numpy(phase))

            if self.return_speed_labels:
                positions = data.get("positions", None)
                if positions is not None:
                    p_slice = positions[start:end]
                    if len(p_slice) > 1:
                        deltas = np.diff(p_slice, axis=0)
                        speeds = np.linalg.norm(deltas, axis=1) / 0.04
                        speeds = np.concatenate([[speeds[0]], speeds])
                    else:
                        speeds = np.zeros(len(p_slice), dtype=np.float32)
                else:
                    speeds = np.zeros(self.seq_len, dtype=np.float32)
                batch_speed.append(torch.from_numpy(speeds.astype(np.float32)))

            if self.return_positions:
                positions = data.get("positions", None)
                if positions is not None:
                    pos_end = end + self.traj_horizon + 1  # +1 for heading computation safety
                    p_slice = positions[start:pos_end]
                else:
                    p_slice = np.zeros((self.seq_len + self.traj_horizon + 1, 2), dtype=np.float32)
                batch_positions.append(torch.from_numpy(p_slice.astype(np.float32)))

        # Stack
        obs = torch.stack([b[:-1] for b in batch_obs])        # t, ..., t+L-1
        obs_next = torch.stack([b[1:] for b in batch_obs])    # t+1, ..., t+L
        actions = torch.stack(batch_act)
        rewards = torch.stack(batch_rew)
        continues = torch.stack(batch_con)

        result = (obs, obs_next, actions, rewards, continues)
        if self.return_phase_labels:
            result += (torch.stack(batch_phase),)
        if self.return_speed_labels:
            result += (torch.stack(batch_speed),)
        if self.return_positions:
            result += (torch.stack(batch_positions),)
        return result

    def _sample_preloaded(self, batch_size):
        """Fast sample from preloaded in-memory data. Normalize on-the-fly."""
        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_con = []
        batch_phase = []
        batch_speed = []
        batch_positions = []

        min_preload_frames = self.seq_len + 1
        if self.return_positions:
            min_preload_frames = max(min_preload_frames, self.seq_len + self.traj_horizon + 1)

        n = len(self._preloaded)
        for _ in range(batch_size):
            fidx = np.random.randint(0, n)
            ep = self._preloaded[fidx]
            T = self._preload_lengths[fidx]
            start = np.random.randint(0, T - min_preload_frames)
            end = start + self.seq_len

            bev_seq = ep["bev"][start:end + 1]  # uint8
            act_seq = torch.from_numpy(ep["actions"][start:end])
            rew_seq = torch.from_numpy(ep["rewards"][start:end])
            don_seq = ep["dones"][start:end]

            batch_obs.append(bev_seq)
            batch_act.append(act_seq)
            batch_rew.append(rew_seq)
            batch_con.append(torch.from_numpy(~don_seq).float())

            if self.return_phase_labels:
                mf = ep.get("merge_frame_idx", -1)
                frames = np.arange(start, end)
                phase = np.zeros(self.seq_len, dtype=np.int64)
                if mf >= 0:
                    phase[frames >= mf] = 1                         # merge-zone
                    phase[frames >= mf + self.merge_zone_frames] = 2  # post-merge (applied after, not overwritten)
                batch_phase.append(torch.from_numpy(phase))

            if self.return_speed_labels:
                speeds = ep.get("speeds", None)
                if speeds is not None:
                    batch_speed.append(torch.from_numpy(speeds[start:end].copy()))
                else:
                    batch_speed.append(torch.zeros(self.seq_len))

            if self.return_positions:
                positions = ep.get("positions", None)
                if positions is not None:
                    pos_end = end + self.traj_horizon + 1
                    p_slice = positions[start:pos_end]
                else:
                    p_slice = np.zeros((self.seq_len + self.traj_horizon + 1, 2), dtype=np.float32)
                batch_positions.append(torch.from_numpy(p_slice.copy() if isinstance(p_slice, np.ndarray) else p_slice))

        # Stack then normalize to float [0,1] (single op, fast on CPU)
        obs = torch.stack([b[:-1] for b in batch_obs]).float() / 255.0
        obs_next = torch.stack([b[1:] for b in batch_obs]).float() / 255.0
        actions = torch.stack(batch_act)
        rewards = torch.stack(batch_rew)
        continues = torch.stack(batch_con)

        result = (obs, obs_next, actions, rewards, continues)
        if self.return_phase_labels:
            result += (torch.stack(batch_phase),)
        if self.return_speed_labels:
            result += (torch.stack(batch_speed),)
        if self.return_positions:
            result += (torch.stack(batch_positions),)
        return result

    def sample_start_obs(self, batch_size):
        """Sample starting observations for imagination training.

        Returns:
            (batch, 3, H, W) float [0,1] — single frame per sample.
        """
        obs, _, _, _, _ = self.sample(batch_size)
        return obs[:, 0]  # (batch, 3, H, W)


def load_all_episodes(data_dir, bev_size=64):
    """
    Load all episodes at once into memory (for small datasets or debugging).
    Returns list of (obs, actions, rewards, dones) tensors.
    """
    loader = OfflineDataset(data_dir, bev_size=bev_size, seq_len=1, cache_size=256)
    episodes = []
    for fpath in sorted(loader._file_list):
        try:
            data = dict(np.load(fpath))
            T = data["actions"].shape[0]
            if T < 2:
                continue
            bev_tensor = torch.from_numpy(data["bev_images"]).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
            if bev_tensor.shape[-1] != bev_size:
                bev_tensor = F.interpolate(
                    bev_tensor, size=(bev_size, bev_size),
                    mode='bilinear', align_corners=False
                )
            episodes.append((
                bev_tensor,
                torch.from_numpy(data["actions"]).float(),
                torch.from_numpy(data["rewards"]).float(),
                torch.from_numpy(data["dones"]),
            ))
        except Exception as e:
            print(f"  [WARN] Failed to load {fpath}: {e}")
    return episodes


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/jiojio/metadrive/mirro_data_map/exid_dreamer_data"
    print(f"Testing OfflineDataset: {data_dir}")

    loader = OfflineDataset(data_dir, bev_size=64, seq_len=50)
    obs, obs_next, actions, rewards, continues = loader.sample(4)

    print(f"\nSample shapes:")
    print(f"  obs:       {obs.shape}       (B, seq_len, 3, H, W)")
    print(f"  obs_next:  {obs_next.shape}")
    print(f"  actions:   {actions.shape}   (B, seq_len, 2)")
    print(f"  rewards:   {rewards.shape}        (B, seq_len)")
    print(f"  continues: {continues.shape}      (B, seq_len)")
    print(f"\nReward stats: min={rewards.min():.2f}, max={rewards.max():.2f}, mean={rewards.mean():.2f}")
