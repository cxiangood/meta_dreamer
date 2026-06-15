"""
Build a MetaDrive ScenarioDescription from the Chinese "Highway-merge-in" trajectory release.

Expected layout (folder passed as ``dataset_dir``):

    Highway-merge-in/
        Trajectory.csv
        TrackIDstate.csv
        Trackstate.txt   (optional, human-readable stats only)

**Which files/columns are used**

- ``Trajectory.csv``: ``frameId``, ``trackId``, ``posX``, ``posY``, ``xVelocity``, ``yVelocity``,
  ``width``, ``height`` (bbox), ``laneId`` (for legacy map only). ``Trackstate.txt`` / ``Road.jpg`` are **not** read.
- ``TrackIDstate.csv``: ``trackId``, ``InitialFrame``, ``TotalFrame``, ``VehicleClass``, ``RampVehicle`` (ramp
  track list when building the optional synthetic map).

Coordinates in Trajectory.csv (posX, posY) are treated as a fixed 2D plane (image / fusion space).
Velocities (xVelocity, yVelocity) are m/s in that same frame, consistent with TrackIDstate distance stats.

**Map geometry** (``map_mode``): default ``"slab"`` — per dataset **说明文档** / ``Road.jpg`` custom frame,
**horizontal = localY**, **vertical = localX** (meters): partition **localY** into slabs of ``slab_step`` (default 5m),
per ``laneId`` take **median localX** in each slab, then **affine-fit** those polylines into **posX/posY** (pixels)
so ``map_features`` match replay tracks. If ``localX``/``localY`` are missing, falls back to ``posY``/``posX`` slabs.
``"legacy"`` uses per-lane representative tracks in image space; ``"clean"`` uses PCA main+ramp. Reference:
``docs/reference_merge_map.png``. ``laneId == 4`` is **not** drawn; vehicles on lane 4 still replay.

**Axis convention**: by default we set ``flip_y_axis=True`` so image-like Y-down maps to MetaDrive's ground
plane (Y up). Pass ``flip_y_axis=False`` if your scene looks mirrored.

**Velocity vs. position**: in this release ``xVelocity``/``yVelocity`` are often *not* colinear with
``posX``/``posY`` frame-to-frame motion (mixed units / body-frame components). We take **speed** from the CSV
vector and **direction** from smoothed position differences so heading matches the replayed path.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.math import get_polyline_length, resample_polyline

# Reference layout: lane 4 is an auxiliary strip beside the ramp — no centerline in ``map_features``.
OMIT_MAP_LANE_IDS = frozenset({4})

# Data-driven lane geometry constants (from Trajectory.csv statistics).
LANE_WIDTH_MAIN = 3.65  # main-road lane center-to-center (L1-L2 stable zone)
LANE_WIDTH_RAMP = 3.0   # ramp-to-L2 gap at merge point (localY 0-50)
RAMP_START_LOCAL_Y = 6.0  # approximate ramp (Lane 3) first-appear localY


def _ref_length_m(vehicle_class: str) -> float:
    c = (vehicle_class or "car").lower()
    if "truck" in c:
        return 8.0
    if "bus" in c:
        return 10.0
    return 4.5


def _bbox_to_size_m(width_px: float, height_px: float, vehicle_class: str) -> Tuple[float, float]:
    w = max(float(width_px), 1.0)
    h = max(float(height_px), 1.0)
    ref = _ref_length_m(vehicle_class)
    long_px, short_px = (w, h) if w >= h else (h, w)
    scale = ref / long_px
    length_m = long_px * scale
    width_m = short_px * scale
    return length_m, width_m


def _heading_and_velocity_from_path(
    pos_xy: np.ndarray, vel_csv: np.ndarray, valid: np.ndarray, min_move: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heading (rad) from position motion; speed from CSV velocity magnitude projected onto that heading.

    The Highway-merge-in CSV often has velocity components that are not aligned with pos deltas (different
    semantics / units). Using atan2(vy,vx) makes vehicle mesh face the wrong way in MetaDrive.
    """
    t = pos_xy.shape[0]
    heading = np.zeros(t, dtype=np.float32)
    vel_out = np.zeros((t, 2), dtype=np.float32)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return heading, vel_out

    for k, i in enumerate(idx):
        i = int(i)
        spd = math.hypot(float(vel_csv[i, 0]), float(vel_csv[i, 1]))
        if k == 0 and idx.size == 1:
            h = 0.0
        elif k == 0:
            jn = int(idx[k + 1])
            d = pos_xy[jn] - pos_xy[i]
        elif k == idx.size - 1:
            jp = int(idx[k - 1])
            d = pos_xy[i] - pos_xy[jp]
        else:
            jp, jn = int(idx[k - 1]), int(idx[k + 1])
            d = pos_xy[jn] - pos_xy[jp]

        dn = float(np.hypot(d[0], d[1]))
        if dn > min_move:
            h = math.atan2(float(d[1]), float(d[0]))
        elif k > 0:
            h = float(heading[int(idx[k - 1])])
        else:
            h = 0.0

        heading[i] = h
        vel_out[i, 0] = spd * math.cos(h)
        vel_out[i, 1] = spd * math.sin(h)

    return heading, vel_out


def _dedupe_consecutive_xy(poly: np.ndarray, min_sep: float) -> np.ndarray:
    if poly.shape[0] == 0:
        return poly.astype(np.float32)
    out = [poly[0]]
    for i in range(1, poly.shape[0]):
        d = math.hypot(float(poly[i, 0] - out[-1][0]), float(poly[i, 1] - out[-1][1]))
        if d >= min_sep:
            out.append(poly[i])
    return np.asarray(out, dtype=np.float32)


def _svd_order_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    t_proj = (centered @ vt[0]).flatten()
    order = np.argsort(t_proj)
    return pts[order].astype(np.float32)


def _resample_lane_polyline(seq: np.ndarray, step_m: float = 3.5) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float64)
    if seq.shape[0] < 2:
        return seq.astype(np.float32)
    length = get_polyline_length(seq)
    if length < step_m * 1.5 or seq.shape[0] < 3:
        return seq.astype(np.float32)
    rs = resample_polyline(seq, step_m)
    if rs.shape[0] < 2:
        return seq.astype(np.float32)
    return rs.astype(np.float32)


def _polyline_for_lane_group(g: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Prefer the longest single-track sequence sorted by time (real centerline motion).
    If that is too short, stitch a few dense tracks; last resort: PCA ordering of all samples.
    """
    if g.shape[0] < 6:
        return None
    counts = g.groupby("trackId").size()
    best_tid = int(counts.idxmax())
    tr = g[g["trackId"] == best_tid].sort_values("frameId")
    seq = tr[["posX", "posY"]].to_numpy(dtype=np.float32)
    seq = _dedupe_consecutive_xy(seq, 0.12)
    if seq.shape[0] >= 2 and get_polyline_length(seq) > 2.5:
        return seq

    stitched = []
    for tid in counts.nlargest(5).index.tolist():
        tr = g[g["trackId"] == int(tid)].sort_values("frameId")
        if len(tr) < 2:
            continue
        arr = _dedupe_consecutive_xy(tr[["posX", "posY"]].to_numpy(dtype=np.float32), 0.12)
        if arr.shape[0] >= 2:
            mf = float(tr["frameId"].mean())
            stitched.append((mf, arr))
    if stitched:
        stitched.sort(key=lambda x: x[0])
        merged = np.vstack([x[1] for x in stitched])
        merged = _dedupe_consecutive_xy(merged, 0.45)
        if merged.shape[0] >= 2 and get_polyline_length(merged) > 2.5:
            return merged

    cloud = g[["posX", "posY"]].to_numpy(dtype=np.float32)
    ordered = _svd_order_points(cloud)
    ordered = _dedupe_consecutive_xy(ordered, 0.25)
    if ordered.shape[0] >= 2:
        return ordered
    return None


def build_map_features_from_trajectory_window(window: pd.DataFrame) -> Dict[str, dict]:
    """
    Build ``map_features`` from *all* rows in the ego time window (not only replay vehicles).

    Lanes are inferred from ``laneId`` + trajectories; geometry follows vehicle motion along each lane.
    """
    features: Dict[str, dict] = {}
    if "laneId" not in window.columns:
        return _placeholder_map_features()

    for lid in window["laneId"].dropna().unique():
        if not (lid == lid):
            continue
        try:
            lid_i = int(float(lid))
        except (TypeError, ValueError):
            continue
        if lid_i in OMIT_MAP_LANE_IDS:
            continue
        g = window[window["laneId"] == lid]
        poly = _polyline_for_lane_group(g)
        if poly is None or poly.shape[0] < 2:
            continue
        arc = get_polyline_length(poly)
        if arc < 4.0:
            continue
        poly = _resample_lane_polyline(poly, step_m=3.5)
        features[str(lid_i)] = {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": poly,
        }

    return features if features else _placeholder_map_features()


def _fit_local_xy_to_pos_xy(window: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Least-squares affine map (localX, localY) → (posX, posY) so slab polylines in meter frame align with replay.
    """
    need = ("localX", "localY", "posX", "posY")
    if not all(c in window.columns for c in need):
        return None
    m = window.dropna(subset=list(need))
    if len(m) < 200:
        return None
    L = m[["localX", "localY"]].to_numpy(dtype=np.float64)
    ones = np.ones((len(L), 1), dtype=np.float64)
    X = np.hstack([L, ones])
    px = m["posX"].to_numpy(dtype=np.float64)
    py = m["posY"].to_numpy(dtype=np.float64)
    cx, *_ = np.linalg.lstsq(X, px, rcond=None)
    cy, *_ = np.linalg.lstsq(X, py, rcond=None)
    wmat = np.array([[cx[0], cy[0]], [cx[1], cy[1]]], dtype=np.float64)
    t = np.array([cx[2], cy[2]], dtype=np.float64)
    return wmat, t


def _polylines_local_xy_to_pos(poly: np.ndarray, wmat: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=np.float64)
    return (p @ wmat.T + t).astype(np.float32)


def build_map_features_slab_lane_statistics(
    window: pd.DataFrame,
    slab_axis: Optional[str] = None,
    value_axis: Optional[str] = None,
    slab_step: float = 5.0,
    min_samples: int = 6,
    ramp_lane_ids: Optional[set] = None,
    smooth_bins: int = 5,
    lane_width_main: float = LANE_WIDTH_MAIN,
    add_boundaries: bool = True,
) -> Dict[str, dict]:
    """
    Lane centerlines from **slab statistics** (see ``步骤.md``):

    - If ``slab_axis`` / ``value_axis`` are ``None`` (default): use **localY** (along-road, m) + **localX** (lateral, m)
      when those columns exist — matching **说明文档** / ``Road.jpg`` (**horizontal localY**, **vertical localX**).
      Polylines are then mapped to **posX/posY** via affine fit on the window. Otherwise fall back to
      ``posY`` / ``posX`` (image space, no extra transform).
    - Partition ``slab_axis`` into intervals of length ``slab_step`` (meters for local*, pixels for pos*).
    - In each slab, per ``laneId``: **median** of ``value_axis`` → lateral center; slab midpoint → along-slab coord.
    - Sort along the slab axis, **moving-average smooth**, resample the polyline.

    **Lane 3** is the ramp in the reference diagram; geometry uses the same rule as other lanes.
    Ramp lanes use a reduced ``min_samples`` (max(3, min_samples // 2)) to handle sparse data at the
    diverging end.

    If ``add_boundaries`` is True, also generates ``BOUNDARY_LINE`` polylines between adjacent lanes
    and at the outer edges of the main road / ramp.
    """
    if ramp_lane_ids is None:
        ramp_lane_ids = {3}
    if slab_axis is None and value_axis is None:
        if "localX" in window.columns and "localY" in window.columns:
            slab_axis, value_axis = "localY", "localX"
        else:
            slab_axis, value_axis = "posY", "posX"
    elif slab_axis is None or value_axis is None:
        raise ValueError("slab_axis and value_axis must both be None or both strings")

    use_local_slab = slab_axis in ("localX", "localY") and value_axis in ("localX", "localY")

    if slab_axis not in window.columns or value_axis not in window.columns or "laneId" not in window.columns:
        return _placeholder_map_features()

    w = window.dropna(subset=[slab_axis, value_axis, "laneId"]).copy()
    if len(w) < 80:
        return build_map_features_from_trajectory_window(window)

    svals = w[slab_axis].to_numpy(dtype=np.float64)
    s_min, s_max = float(np.percentile(svals, 0.5)), float(np.percentile(svals, 99.5))
    if s_max - s_min < slab_step * 2.0:
        return build_map_features_from_trajectory_window(window)

    edges = np.arange(math.floor(s_min / slab_step) * slab_step, s_max + slab_step * 1.001, slab_step, dtype=np.float64)
    # Row is [value_median, slab_mid] when slab is *Y (image posY or meter localY); else [slab_mid, value_median].
    sort_dim = 1 if slab_axis in ("posY", "localY") else 0

    # Per-slab per-lane raw centers with envelope (used for centerlines & boundaries).
    # Each entry: (s_mid, c_val, p_lo, p_hi) where p_lo/p_hi are min/max of value_axis.
    raw_centers: Dict[int, List[Tuple[float, float, float, float]]] = {}
    features: Dict[str, dict] = {}

    for lid in w["laneId"].dropna().unique():
        if lid != lid:
            continue
        try:
            lid_i = int(float(lid))
        except (TypeError, ValueError):
            continue
        if lid_i in OMIT_MAP_LANE_IDS:
            continue
        # Ramp lanes get reduced min_samples to handle sparse data at diverging end.
        eff_min = max(3, min_samples // 2) if lid_i in ramp_lane_ids else min_samples
        rows: List[List[float]] = []
        centers_list: List[Tuple[float, float, float, float]] = []
        for i in range(len(edges) - 1):
            s0, s1 = float(edges[i]), float(edges[i + 1])
            s_mid = 0.5 * (s0 + s1)
            m = (w["laneId"] == lid) & (w[slab_axis] >= s0) & (w[slab_axis] < s1)
            vals = w.loc[m, value_axis].to_numpy(dtype=np.float64)
            if vals.size < eff_min:
                continue
            c_val = float(np.median(vals))
            p_lo, p_hi = float(np.min(vals)), float(np.max(vals))
            if slab_axis in ("posY", "localY"):
                rows.append([c_val, s_mid])
            else:
                rows.append([s_mid, c_val])
            centers_list.append((s_mid, c_val, p_lo, p_hi))

        if len(rows) < 4:
            continue
        raw_centers[lid_i] = centers_list

        poly = np.asarray(rows, dtype=np.float64)
        poly = poly[np.argsort(poly[:, sort_dim])]
        sw = min(max(3, smooth_bins | 1), poly.shape[0] if poly.shape[0] % 2 == 1 else poly.shape[0] - 1)
        poly = _moving_average_2d(poly.astype(np.float32), sw)
        arc = get_polyline_length(poly)
        if arc < 6.0:
            continue
        poly = _resample_lane_polyline(poly.astype(np.float64), 3.5).astype(np.float32)
        features[str(lid_i)] = {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": poly,
        }

    if len(features) < 1:
        return build_map_features_from_trajectory_window(window)

    # Merge ramp into adjacent main lane centerline (by laneId).
    main_ids = sorted(set(features.keys()) - {str(r) for r in ramp_lane_ids})
    ramp_ids_str = [str(r) for r in sorted(ramp_lane_ids) if str(r) in features]
    if ramp_ids_str and main_ids:
        ramp_key = ramp_ids_str[0]
        ramp_id_int = int(ramp_key)
        # Pick main lane whose laneId is closest to (but not equal to) the ramp's.
        candidates = [(abs(int(m) - ramp_id_int), m) for m in main_ids if int(m) != ramp_id_int]
        if candidates:
            candidates.sort()
            best_main = candidates[0][1]
            features[ramp_key]["polyline"] = _merge_ramp_into_lane(
                features[ramp_key]["polyline"], features[best_main]["polyline"]
            )

    # Boundary lines between adjacent lanes and outer edges.
    if add_boundaries and len(raw_centers) >= 2:
        features = _add_slab_boundaries(features, raw_centers, sort_dim)

    if use_local_slab:
        ft = _fit_local_xy_to_pos_xy(window)
        if ft is None:
            return build_map_features_slab_lane_statistics(
                window,
                slab_axis="posY",
                value_axis="posX",
                slab_step=slab_step,
                min_samples=min_samples,
                ramp_lane_ids=ramp_lane_ids,
                smooth_bins=smooth_bins,
                lane_width_main=lane_width_main,
                add_boundaries=add_boundaries,
            )
        wmat, t = ft
        for data in features.values():
            data["polyline"] = _polylines_local_xy_to_pos(data["polyline"], wmat, t)

    return features


def _add_slab_boundaries(
    features: Dict[str, dict],
    raw_centers: Dict[int, List[Tuple[float, float, float, float]]],
    sort_dim: int,
) -> Dict[str, dict]:
    """
    Build ``BOUNDARY_LINE`` polylines from slab-level envelope data.

    - Between each pair of adjacent lanes: midpoint between inner envelope edges
      (la's p_lo and lb's p_hi), ensuring the boundary sits between the actual
      trajectory extents of the two lanes.
    - Outer edges: actual envelope edges (p_hi for rightmost lane, p_lo for
      leftmost lane) so boundaries contain ALL trajectories.
    """
    sorted_lids = sorted(raw_centers.keys())

    # Inter-lane boundary: midpoint between inner envelope edges, slab-by-slab.
    for idx in range(len(sorted_lids) - 1):
        la, lb = sorted_lids[idx], sorted_lids[idx + 1]
        # Each entry: (s_mid, c_val, p_lo, p_hi)
        ca = {t[0]: t for t in raw_centers[la]}
        cb = {t[0]: t for t in raw_centers[lb]}
        common = sorted(set(ca.keys()) & set(cb.keys()))
        if len(common) < 4:
            continue
        rows: List[List[float]] = []
        for s_mid in common:
            # la's p_lo (most negative edge, towards lb) and lb's p_hi (least negative, towards la)
            mid_val = 0.5 * (ca[s_mid][2] + cb[s_mid][3])
            if sort_dim == 1:  # slab is Y
                rows.append([mid_val, s_mid])
            else:
                rows.append([s_mid, mid_val])
        poly = np.asarray(rows, dtype=np.float64)
        poly = poly[np.argsort(poly[:, sort_dim])]
        poly = _moving_average_2d(poly.astype(np.float32), 5)
        arc = get_polyline_length(poly)
        if arc < 6.0:
            continue
        poly = _resample_lane_polyline(poly.astype(np.float64), 3.5).astype(np.float32)
        features[f"boundary_{la}_{lb}"] = {
            "type": MetaDriveType.BOUNDARY_LINE,
            "polyline": poly,
        }

    # Outer edges: use actual envelope edges of outermost lanes.
    # sorted_lids[0] is the least-negative-localX lane (rightmost in image);
    #   its outer edge = p_hi (least negative, extending away from center).
    # sorted_lids[-1] is the most-negative-localX lane (leftmost in image);
    #   its outer edge = p_lo (most negative, extending away from center).
    for lid, edge_idx in [(sorted_lids[0], 3), (sorted_lids[-1], 2)]:
        centers = raw_centers[lid]
        if len(centers) < 4:
            continue
        rows = []
        for entry in centers:
            s_mid = entry[0]
            edge_val = entry[edge_idx]
            if sort_dim == 1:
                rows.append([edge_val, s_mid])
            else:
                rows.append([s_mid, edge_val])
        poly = np.asarray(rows, dtype=np.float64)
        poly = poly[np.argsort(poly[:, sort_dim])]
        poly = _moving_average_2d(poly.astype(np.float32), 5)
        arc = get_polyline_length(poly)
        if arc < 6.0:
            continue
        poly = _resample_lane_polyline(poly.astype(np.float64), 3.5).astype(np.float32)
        side = "R" if edge_idx == 3 else "L"
        features[f"edge_{side}"] = {
            "type": MetaDriveType.BOUNDARY_LINE,
            "polyline": poly,
        }

    return features


def _placeholder_map_features() -> Dict[str, dict]:
    return {
        "lane_placeholder": {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        }
    }


def _moving_average_2d(poly: np.ndarray, win: int) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64)
    n = poly.shape[0]
    if n < 3 or win < 3:
        return poly.astype(np.float32)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 3:
        return poly.astype(np.float32)
    if win % 2 == 0:
        win -= 1
    pad = win // 2
    k = np.ones(win, dtype=np.float64) / win
    px = np.pad(poly[:, 0], (pad, pad), mode="edge")
    py = np.pad(poly[:, 1], (pad, pad), mode="edge")
    sx = np.convolve(px, k, mode="valid")
    sy = np.convolve(py, k, mode="valid")
    return np.stack([sx, sy], axis=1).astype(np.float32)


def _binned_median_spine(pts_xy: np.ndarray, n_bins: int) -> Optional[np.ndarray]:
    """1D binning along first PCA axis → smooth road spine in (x,y) space."""
    pts = np.asarray(pts_xy, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 50:
        return None
    for j in range(2):
        lo, hi = np.percentile(pts[:, j], [0.2, 99.8])
        pts = pts[(pts[:, j] >= lo) & (pts[:, j] <= hi)]
    if pts.shape[0] < 50:
        return None
    if np.std(pts[:, 0]) < 1e-3 and np.std(pts[:, 1]) < 1e-3:
        return None
    if pts.shape[0] > 22000:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=22000, replace=False)
        pts = pts[idx]
    c = pts.mean(axis=0)
    x = (pts - c).astype(np.float64)
    try:
        _, s_sing, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(s_sing).all() or float(s_sing[0]) < 1e-8:
        return None
    u = vt[0] / (float(np.linalg.norm(vt[0])) + 1e-9)
    if not np.all(np.isfinite(u)):
        return None
    s = x[:, 0] * u[0] + x[:, 1] * u[1]
    s_lo, s_hi = float(np.percentile(s, 0.5)), float(np.percentile(s, 99.5))
    if s_hi - s_lo < 8.0:
        return None
    edges = np.linspace(s_lo, s_hi, n_bins + 1)
    centers: List[List[float]] = []
    for i in range(n_bins):
        m = (s >= edges[i]) & (s < edges[i + 1])
        if np.count_nonzero(m) < 2:
            centers.append([float("nan"), float("nan")])
        else:
            centers.append([float(np.median(pts[m, 0])), float(np.median(pts[m, 1]))])
    poly = np.asarray(centers, dtype=np.float64)
    for j in range(2):
        col = poly[:, j]
        mask = ~np.isfinite(col)
        if not mask.any():
            continue
        ix = np.arange(poly.shape[0])
        good = ~mask
        if np.count_nonzero(good) < 2:
            return None
        col[mask] = np.interp(ix[mask], ix[good], col[good])
    poly = poly.astype(np.float32)
    poly = _dedupe_consecutive_xy(poly, 0.05)
    if poly.shape[0] < 10:
        return None
    return poly


def _offset_lane_centerline(poly: np.ndarray, offset: float) -> np.ndarray:
    """Offset each point perpendicular to local tangent (left = +offset)."""
    p = np.asarray(poly, dtype=np.float64)
    n = p.shape[0]
    out = np.zeros_like(p)
    for i in range(n):
        i0, i2 = max(0, i - 1), min(n - 1, i + 1)
        t = p[i2] - p[i0]
        tn = float(np.linalg.norm(t))
        if tn < 1e-6:
            t = np.array([1.0, 0.0])
        else:
            t = t / tn
        normal = np.array([-t[1], t[0]])
        out[i] = p[i] + offset * normal
    return out.astype(np.float32)


def _merge_ramp_into_lane(ramp: np.ndarray, main_lane: np.ndarray, bridge_pts: int = 14) -> np.ndarray:
    """Extend ramp polyline to connect into the main lane at the merge end.

    Finds where the ramp's *last* point is closest to the main lane, then appends
    a bridge + tail of the main lane so the ramp visually merges into it.
    """
    r = np.asarray(ramp, dtype=np.float64)
    m = np.asarray(main_lane, dtype=np.float64)
    if r.shape[0] < 2 or m.shape[0] < 2:
        return ramp.astype(np.float32)

    # Find main-lane point closest to the ramp's last point (merge end).
    d2 = np.sum((m - r[-1]) ** 2, axis=1)
    jm = int(np.argmin(d2))

    # Bridge from ramp end to that main-lane point.
    bridge = np.linspace(r[-1], m[jm], num=bridge_pts, dtype=np.float64)[1:]  # skip ramp[-1] duplicate

    # Tail: remaining main-lane points after the merge junction.
    n_tail = min(28, m.shape[0] - jm - 1)
    tail = m[jm + 1: jm + 1 + n_tail].astype(np.float64) if n_tail > 0 else np.empty((0, 2), dtype=np.float64)

    parts = [r, bridge]
    if tail.shape[0] > 0:
        parts.append(tail)
    out = np.vstack(parts)
    out = _dedupe_consecutive_xy(out.astype(np.float32), 0.08)
    return out


def _snap_ramp_to_mainline(ramp: np.ndarray, main_center: np.ndarray, bridge_pts: int = 18) -> np.ndarray:
    """Truncate ramp at closest approach to main center, then linearly tie into that point."""
    r = np.asarray(ramp, dtype=np.float64)
    m = np.asarray(main_center, dtype=np.float64)
    if r.shape[0] < 3 or m.shape[0] < 3:
        return ramp.astype(np.float32)
    best_d, ir, jm = 1e30, 0, 0
    for i in range(r.shape[0]):
        d2 = np.sum((m - r[i]) ** 2, axis=1)
        j = int(np.argmin(d2))
        dd = float(d2[j])
        if dd < best_d:
            best_d, ir, jm = dd, i, j
    head = r[: ir + 1]
    bridge = np.linspace(r[ir], m[jm], num=max(bridge_pts, 6), dtype=np.float64)
    if head.shape[0] > 1:
        out = np.vstack([head[:-1], bridge[1:]])
    else:
        out = bridge.astype(np.float64)
    n_tail = min(28, m.shape[0] - jm - 1)
    if n_tail > 2:
        tail = m[jm + 1 : jm + 1 + n_tail : 2]
        out = np.vstack([out, tail.astype(np.float64)])
    out = _dedupe_consecutive_xy(out.astype(np.float32), 0.08)
    return out


def build_clean_merge_map_features(
    window: pd.DataFrame,
    meta: pd.DataFrame,
    lane_width: float = LANE_WIDTH_MAIN,
    main_bins: int = 110,
    ramp_bins: int = 42,
) -> Dict[str, dict]:
    """
    Synthetic but *smooth* highway map: one main spine from all traffic, parallel main lanes,
    a ramp spine from ramp-flagged tracks merged into the main centerline with a short blend.
    Also adds outer ``ROAD_EDGE_BOUNDARY`` polylines so edges are straight-ish, not jagged laneId polylines.
    """
    pts = window[["posX", "posY"]].dropna().to_numpy(dtype=np.float64)
    spine = _binned_median_spine(pts, main_bins)
    if spine is None:
        return build_map_features_from_trajectory_window(window)

    sm_win = min(27, max(7, spine.shape[0] // 4 * 2 + 1))
    spine = _moving_average_2d(spine, sm_win)
    spine = _resample_lane_polyline(spine.astype(np.float64), 3.8).astype(np.float32)

    features: Dict[str, dict] = {}
    offsets = [-lane_width, 0.0, lane_width]
    for li, off in enumerate(offsets):
        pl = _offset_lane_centerline(spine, float(off))
        if pl.shape[0] >= 2 and get_polyline_length(pl) > 5.0:
            features[f"main_L{li}"] = {
                "type": MetaDriveType.LANE_SURFACE_STREET,
                "polyline": pl,
            }

    main_center = _offset_lane_centerline(spine, 0.0)
    outer = 1.65 * lane_width
    el = _offset_lane_centerline(spine, -outer)
    er = _offset_lane_centerline(spine, outer)
    if el.shape[0] >= 2:
        features["edge_L"] = {"type": MetaDriveType.BOUNDARY_LINE, "polyline": el}
    if er.shape[0] >= 2:
        features["edge_R"] = {"type": MetaDriveType.BOUNDARY_LINE, "polyline": er}

    ramp_ids = meta.loc[meta["RampVehicle"] == True, "trackId"].astype(int).unique().tolist()
    present = set(int(t) for t in window["trackId"].unique())
    ramp_ids = [tid for tid in ramp_ids if tid in present]
    if ramp_ids:
        rw = window[window["trackId"].isin(ramp_ids)]
        rpts = rw[["posX", "posY"]].dropna().to_numpy(dtype=np.float64)
        if rpts.shape[0] > 80:
            r_spine = _binned_median_spine(rpts, ramp_bins)
            if r_spine is not None and r_spine.shape[0] >= 6:
                rwin = min(17, max(5, r_spine.shape[0] // 3 * 2 + 1))
                r_spine = _moving_average_2d(r_spine, rwin)
                r_spine = _resample_lane_polyline(r_spine.astype(np.float64), 3.2).astype(np.float32)
                r_merged = _snap_ramp_to_mainline(r_spine, main_center.astype(np.float64))
                if r_merged.shape[0] >= 4 and get_polyline_length(r_merged) > 8.0:
                    features["ramp"] = {
                        "type": MetaDriveType.LANE_SURFACE_STREET,
                        "polyline": r_merged.astype(np.float32),
                    }

    return features if len(features) >= 2 else build_map_features_from_trajectory_window(window)


def build_highway_merge_scenario(
    dataset_dir: str,
    sdc_track_id: Optional[int] = None,
    max_traffic: int = 48,
    prefer_ramp_ego: bool = True,
    flip_y_axis: bool = True,
    clean_map: bool = False,
    map_mode: str = "slab",
) -> SD:
    """
    Args:
        dataset_dir: Directory that contains ``Trajectory.csv`` and ``TrackIDstate.csv``.
        sdc_track_id: Ego track id; if None, picks first ramp vehicle when ``prefer_ramp_ego`` else first track.
        max_traffic: Max number of *other* vehicles besides ego (min distance over co-visible frames).
        prefer_ramp_ego: If True and ``sdc_track_id`` is None, use first ``RampVehicle`` in TrackIDstate.
        flip_y_axis: If True (default), negate ``posY`` and ``yVelocity`` for typical BEV image Y-down data.
        clean_map: If True, force ``map_mode="clean"`` (synthetic PCA map).
        map_mode: ``"slab"`` (default, localY-slab + laneId localX median → affine to pos), ``"legacy"``, ``"clean"``.

    Returns:
        ``ScenarioDescription`` instance (pass directly to ``ScenarioOnlineEnv.set_scenario``; the engine
        will centralize to ego and run ``sanity_check`` again).
    """
    traj_path = os.path.join(dataset_dir, "Trajectory.csv")
    meta_path = os.path.join(dataset_dir, "TrackIDstate.csv")
    if not os.path.isfile(traj_path) or not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Need Trajectory.csv and TrackIDstate.csv under {dataset_dir}")

    meta = pd.read_csv(meta_path)
    if sdc_track_id is None:
        if prefer_ramp_ego and meta["RampVehicle"].any():
            sdc_track_id = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0])
        else:
            sdc_track_id = int(meta["trackId"].iloc[0])

    ego_row = meta.loc[meta["trackId"] == sdc_track_id]
    if ego_row.empty:
        raise ValueError(f"trackId {sdc_track_id} not found in TrackIDstate.csv")
    ego_row = ego_row.iloc[0]
    f0 = int(ego_row["InitialFrame"])
    n_frames = int(ego_row["TotalFrame"])
    f1 = f0 + n_frames - 1
    ego_class = str(ego_row["VehicleClass"])

    traj = pd.read_csv(traj_path)
    traj.columns = [c.strip() for c in traj.columns]

    window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()
    if window.empty:
        raise ValueError(f"No trajectory rows for ego window [{f0}, {f1}]")
    if flip_y_axis:
        window["posY"] = -window["posY"]
        window["yVelocity"] = -window["yVelocity"]

    ego_rows = window[window["trackId"] == sdc_track_id][["frameId", "posX", "posY"]]
    ego_by_f = {int(float(r["frameId"])): (float(r["posX"]), float(r["posY"])) for _, r in ego_rows.iterrows()}

    candidates = window["trackId"].unique()
    dists = []
    for tid in candidates:
        if int(tid) == sdc_track_id:
            continue
        sub = window[window["trackId"] == tid][["frameId", "posX", "posY"]]
        best = None
        for _, r in sub.iterrows():
            fi = int(float(r["frameId"]))
            if fi not in ego_by_f:
                continue
            ex, ey = ego_by_f[fi]
            d = math.hypot(float(r["posX"]) - ex, float(r["posY"]) - ey)
            if best is None or d < best:
                best = d
        if best is not None:
            dists.append((best, int(tid)))
    dists.sort(key=lambda x: x[0])
    other_ids = [tid for _, tid in dists[:max_traffic]]
    selected_ids: List[int] = [sdc_track_id] + other_ids

    meta_by_id = meta.set_index("trackId")
    t_len = f1 - f0 + 1
    ts = (np.arange(t_len, dtype=np.float32) * 0.1)

    tracks = {}

    for tid in selected_ids:
        try:
            vclass = str(meta_by_id.loc[tid, "VehicleClass"])
        except KeyError:
            vclass = "car"
        sub = window[window["trackId"] == tid].sort_values("frameId")
        pos = np.zeros((t_len, 3), dtype=np.float32)
        vel = np.zeros((t_len, 2), dtype=np.float32)
        valid = np.zeros(t_len, dtype=bool)
        length = np.zeros(t_len, dtype=np.float32)
        width = np.zeros(t_len, dtype=np.float32)
        height = np.ones(t_len, dtype=np.float32) * 1.5

        for _, r in sub.iterrows():
            fi = int(float(r["frameId"])) - f0
            if fi < 0 or fi >= t_len:
                continue
            pos[fi, 0] = float(r["posX"])
            pos[fi, 1] = float(r["posY"])
            pos[fi, 2] = 0.0
            vel[fi, 0] = float(r["xVelocity"])
            vel[fi, 1] = float(r["yVelocity"])
            valid[fi] = True
            lm, wm = _bbox_to_size_m(float(r["width"]), float(r["height"]), vclass)
            length[fi] = lm
            width[fi] = wm

        if not valid.any():
            continue
        first_valid = int(np.argmax(valid))
        default_len = float(length[first_valid]) if length[first_valid] > 1e-3 else _ref_length_m(vclass)
        default_wid = float(width[first_valid]) if width[first_valid] > 1e-3 else default_len * 0.42
        for i in range(t_len):
            if not valid[i]:
                length[i] = 0.0
                width[i] = 0.0
                continue
            if length[i] < 1e-3:
                length[i] = default_len
            if width[i] < 1e-3:
                width[i] = default_wid
        heading, vel = _heading_and_velocity_from_path(pos[:, :2], vel, valid)
        tid_str = str(tid)
        tracks[tid_str] = {
            "type": MetaDriveType.VEHICLE,
            "state": {
                "position": pos,
                "velocity": vel,
                "heading": heading,
                "valid": valid,
                "length": length,
                "width": width,
                "height": height,
            },
            "metadata": {
                "type": MetaDriveType.VEHICLE,
                "object_id": tid_str,
                "dataset": "highway_merge_in",
            },
        }

    sdc_str = str(sdc_track_id)
    if sdc_str not in tracks:
        raise RuntimeError("Ego track missing after filtering; check Trajectory.csv / track id.")

    mode = "clean" if clean_map else map_mode
    if mode == "clean":
        map_features = build_clean_merge_map_features(window, meta)
    elif mode == "legacy":
        map_features = build_map_features_from_trajectory_window(window)
    else:
        map_features = build_map_features_slab_lane_statistics(window)

    scenario = {
        "id": f"highway-merge-in-{sdc_track_id}",
        "version": "MetaDrive v0.3.0.1",
        "length": t_len,
        "metadata": {
            "metadrive_processed": True,
            "coordinate": MetaDriveType.COORDINATE_METADRIVE,
            "ts": ts,
            "sdc_id": sdc_str,
            "scenario_id": f"hmi_{sdc_track_id}_{f0}_{f1}",
            "dataset": "highway_merge_in",
            "source_file": os.path.basename(dataset_dir),
            "ego_vehicle_class": ego_class,
            "frame_range": (f0, f1),
        },
        "tracks": tracks,
        "dynamic_map_states": {},
        "map_features": map_features,
    }
    SD.sanity_check(scenario, check_self_type=True)
    return SD(scenario)


def default_dataset_dir() -> str:
    """Default extract path used by the example script (zip extracted without PDF)."""
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "third_party_data",
        "highway_merge_in",
        "Highway-merge-in",
    )
    return root
