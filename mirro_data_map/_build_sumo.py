"""
Step 1: Generate SUMO .net.xml for highway-merge-in
Step 2: Build scenario with localXY coordinates (no affine needed)
Step 3: Render BEV in MetaDrive
"""
import os, sys, math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# ─── Config ───
DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "third_party_data", "highway_merge_in")
os.makedirs(OUTPUT_DIR, exist_ok=True)
NET_XML_PATH = os.path.join(OUTPUT_DIR, "highway_merge.net.xml")

LANE_WIDTH_MAIN = 3.75   # standard highway lane width (m)
LANE_WIDTH_RAMP = 3.0    # ramp lane width (m)
SPEED_MAIN = 33.33       # 120 km/h in m/s
SPEED_RAMP = 16.67       # 60 km/h in m/s

# ─── Data analysis: extract ramp geometry ───
print("=== Analyzing trajectory data ===")
traj = pd.read_csv(os.path.join(DATASET_DIR, "Trajectory.csv"))
traj.columns = [c.strip() for c in traj.columns]
meta = pd.read_csv(os.path.join(DATASET_DIR, "TrackIDstate.csv"))

# Lane centers in localX
for lid in [1, 2, 3]:
    sub = traj[traj["laneId"] == lid]
    print(f"  Lane {lid}: localX median={sub.localX.median():.2f}  "
          f"localY range=[{sub.localY.min():.1f}, {sub.localY.max():.1f}]")

# Get ramp centerline (Lane 3) per localY slab
ramp_data = traj[traj["laneId"] == 3].dropna(subset=["localX", "localY"])
slab_step = 10.0
ly_vals = ramp_data["localY"].values
ly_min, ly_max = np.percentile(ly_vals, 1), np.percentile(ly_vals, 99)
edges = np.arange(math.floor(ly_min / slab_step) * slab_step, ly_max + slab_step, slab_step)
ramp_center = []
for i in range(len(edges) - 1):
    m = (ramp_data["localY"] >= edges[i]) & (ramp_data["localY"] < edges[i+1])
    if m.sum() < 5:
        continue
    lx_median = ramp_data.loc[m, "localX"].median()
    ly_mid = 0.5 * (edges[i] + edges[i+1])
    ramp_center.append((ly_mid, lx_median))
ramp_center = np.array(ramp_center)
print(f"  Ramp centerline: {len(ramp_center)} slab points")
print(f"    start: localY={ramp_center[0][0]:.1f}, localX={ramp_center[0][1]:.2f}")
print(f"    end:   localY={ramp_center[-1][0]:.1f}, localX={ramp_center[-1][1]:.2f}")

# ─── Generate .net.xml ───
# SUMO coords: X = localY (along road), Y = localX (lateral)
# Reference line at Y = 0 (localX = 0), lanes to the RIGHT (negative localX)

# Lane positions with 3.75m width
lane1_center_y = -LANE_WIDTH_MAIN / 2                    # -1.875
lane2_center_y = -(LANE_WIDTH_MAIN + LANE_WIDTH_MAIN/2)  # -5.625

# Road extent
road_y_start = -10.0
road_y_end = 220.0

# ─── Ramp geometry ───
# Lane 3 data only covers the parallel portion (vehicles still labeled laneId=3).
# The actual merge curve must be defined based on physical road layout.
#
# Ramp runs parallel at localX ≈ -8.64 from localY ≈ 6 to localY ≈ 120,
# then curves toward Lane 2 over ~50m, merging at localY ≈ 160.

# Use the data slab centerline for the parallel portion
ramp_parallel = ramp_center[ramp_center[:, 0] <= 120.0]  # parallel section
if len(ramp_parallel) < 2:
    ramp_parallel = ramp_center[:3]

# Ramp localX center from data (use median of parallel portion)
ramp_lx_center = float(np.median(ramp_parallel[:, 1]))
ramp_start_ly = float(ramp_parallel[0, 0])

# Merge junction position
merge_localY = 160.0
merge_node_y = lane2_center_y  # connect at Lane 2 center

# Build ramp shape: parallel section + smooth merge curve
ramp_shape_pts = []
# Parallel section from data
for ly, lx in ramp_parallel:
    ramp_shape_pts.append((ly, lx))

# Transition curve: from last parallel point to merge at Lane 2
if len(ramp_shape_pts) > 0:
    last_ly, last_lx = ramp_shape_pts[-1]
else:
    last_ly, last_lx = ramp_start_ly, ramp_lx_center

# Smooth cubic transition from (last_ly, last_lx) to (merge_localY, lane2_center_y)
n_curve = 10
for i in range(1, n_curve + 1):
    t = i / n_curve
    # Smooth easing (ease-in-out)
    t_smooth = 3*t**2 - 2*t**3
    ly = last_ly + t_smooth * (merge_localY - last_ly)
    lx = last_lx + t_smooth * (lane2_center_y - last_lx)
    ramp_shape_pts.append((ly, lx))

ramp_shape = " ".join(f"{ly:.2f},{lx:.2f}" for ly, lx in ramp_shape_pts)

print(f"  Ramp: {len(ramp_parallel)} parallel pts + {n_curve} curve pts")
print(f"  Parallel: localY=[{ramp_parallel[0][0]:.0f}, {ramp_parallel[-1][0]:.0f}], "
      f"localX median={ramp_lx_center:.2f}")
print(f"  Merge at: localY={merge_localY:.0f}, localX={merge_node_y:.2f}")

# Nodes
nodes = [
    ("main_start", road_y_start, 0.0, "priority"),
    ("main_end", road_y_end, 0.0, "priority"),
    ("ramp_start", ramp_start_ly, ramp_lx_center + LANE_WIDTH_RAMP/2, "priority"),
]
nodes.append(("merge", merge_localY, merge_node_y, "zipper"))

print(f"\n=== Generating .net.xml ===")
print(f"  Nodes:")
for nid, x, y, nt in nodes:
    print(f"    {nid}: ({x:.1f}, {y:.2f}) type={nt}")

# Build XML
root = ET.Element("net", version="1.9")
loc = ET.SubElement(root, "location",
                    netOffset="0,0",
                    convBoundary=f"{road_y_start},{lane2_center_y - 5},{road_y_end},{LANE_WIDTH_MAIN + 1}",
                    origBoundary=f"{road_y_start},{lane2_center_y - 5},{road_y_end},{LANE_WIDTH_MAIN + 1}")

# Nodes
for nid, x, y, nt in nodes:
    ET.SubElement(root, "node", id=nid, x=f"{x:.2f}", y=f"{y:.2f}", type=nt)

# Edges
def add_edge(edge_id, from_node, to_node, n_lanes, width, speed, shape=None):
    attrs = {
        "id": edge_id, "from": from_node, "to": to_node,
        "priority": "3", "numLanes": str(n_lanes),
        "speed": f"{speed:.2f}",
    }
    if shape:
        attrs["shape"] = shape
    edge = ET.SubElement(root, "edge", **attrs)
    for i in range(n_lanes):
        ET.SubElement(edge, "lane",
                      id=f"{edge_id}_{i}", index=str(i),
                      speed=f"{speed:.2f}", width=f"{width:.2f}")

# Main road before merge
add_edge("main_pre", "main_start", "merge", 2, LANE_WIDTH_MAIN, SPEED_MAIN)

# Main road after merge
add_edge("main_post", "merge", "main_end", 2, LANE_WIDTH_MAIN, SPEED_MAIN)

# Ramp
add_edge("ramp", "ramp_start", "merge", 1, LANE_WIDTH_RAMP, SPEED_RAMP, shape=ramp_shape)

# Connection: ramp lane 0 → main_post lane 1 (outer lane = Lane 2)
ET.SubElement(root, "connection",
              **{"from": "ramp", "to": "main_post",
                 "fromLane": "0", "toLane": "1", "pass": "1"})

# Write file
tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write(NET_XML_PATH, encoding="UTF-8", xml_declaration=True)
print(f"\n  Saved → {NET_XML_PATH}")
print(f"  Ramp shape: {len(ramp_shape_pts)} points")

# ─── Build map_features from .net.xml geometry ───
def build_map_features_from_net_xml():
    """Parse the .net.xml and extract lane centerline polylines in localXY."""
    features = {}

    # Main road lanes (straight lines in localY direction)
    for lane_id, center_y, label in [
        ("1", lane1_center_y, "Lane 1 (inner)"),
        ("2", lane2_center_y, "Lane 2 (outer)"),
    ]:
        # Centerline from road_start to road_end
        pts = np.array([
            [road_y_start, center_y],
            [road_y_end, center_y]
        ], dtype=np.float32)
        features[lane_id] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": pts,
        }
        print(f"  {label}: ({road_y_start}, {center_y:.2f}) → ({road_y_end}, {center_y:.2f})")

    # Ramp centerline from shape points (already tuples of (localY, localX))
    ramp_pts = np.array(ramp_shape_pts, dtype=np.float32)
    features["3"] = {
        "type": "LANE_SURFACE_STREET",
        "polyline": ramp_pts,
    }
    print(f"  Ramp: {len(ramp_pts)} points, start=({ramp_pts[0][0]:.1f},{ramp_pts[0][1]:.2f}) "
          f"end=({ramp_pts[-1][0]:.1f},{ramp_pts[-1][1]:.2f})")

    # Lane boundaries
    boundary_y_12 = -(LANE_WIDTH_MAIN)  # -3.75 (boundary between Lane 1 and Lane 2)
    features["boundary_1_2"] = {
        "type": "ROAD_EDGE_BOUNDARY",
        "polyline": np.array([[road_y_start, boundary_y_12], [road_y_end, boundary_y_12]], dtype=np.float32),
    }
    # Outer edges
    features["edge_L"] = {
        "type": "ROAD_EDGE_BOUNDARY",
        "polyline": np.array([[road_y_start, 0.0], [road_y_end, 0.0]], dtype=np.float32),
    }
    features["edge_R"] = {
        "type": "ROAD_EDGE_BOUNDARY",
        "polyline": np.array([[road_y_start, lane2_center_y - LANE_WIDTH_MAIN/2],
                              [road_y_end, lane2_center_y - LANE_WIDTH_MAIN/2]], dtype=np.float32),
    }
    print(f"  Boundary 1-2: Y={boundary_y_12:.2f}")
    print(f"  Edge L: Y=0.0")
    print(f"  Edge R: Y={lane2_center_y - LANE_WIDTH_MAIN/2:.2f}")

    return features

print(f"\n=== Building map_features ===")
map_features = build_map_features_from_net_xml()

# ─── Build scenario with localXY ───
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.math import get_polyline_length

# Use first ramp vehicle as ego
if meta["RampVehicle"].any():
    sdc_track_id = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0])
else:
    sdc_track_id = int(meta["trackId"].iloc[0])

ego_row = meta.loc[meta["trackId"] == sdc_track_id].iloc[0]
f0 = int(ego_row["InitialFrame"])
n_frames = int(ego_row["TotalFrame"])
f1 = f0 + n_frames - 1
ego_class = str(ego_row["VehicleClass"])

window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()

# Select traffic vehicles
candidates = window["trackId"].unique()
other_ids = [int(t) for t in candidates if int(t) != sdc_track_id][:48]
selected_ids = [sdc_track_id] + other_ids

meta_by_id = meta.set_index("trackId")
t_len = f1 - f0 + 1
ts = np.arange(t_len, dtype=np.float32) * 0.1

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
        # Use localXY (meters): X = localY, Y = localX
        pos[fi, 0] = float(r["localY"])
        pos[fi, 1] = float(r["localX"])
        pos[fi, 2] = 0.0
        vel[fi, 0] = float(r["xVelocity"])
        vel[fi, 1] = float(r["yVelocity"])
        valid[fi] = True
        w, h = float(r["width"]), float(r["height"])
        ref_len = 4.5 if "truck" not in vclass.lower() else 8.0
        scale = ref_len / max(w, h, 1.0)
        length[fi] = max(w, h) * scale
        width[fi] = min(w, h) * scale

    if not valid.any():
        continue

    # Heading from position diffs in localXY
    heading = np.zeros(t_len, dtype=np.float32)
    idx = np.flatnonzero(valid)
    for k, i in enumerate(idx):
        i = int(i)
        if k == 0 and idx.size == 1:
            h = 0.0
        elif k == 0:
            jn = int(idx[k + 1])
            d = pos[jn] - pos[i]
        elif k == idx.size - 1:
            jp = int(idx[k - 1])
            d = pos[i] - pos[jp]
        else:
            jp, jn = int(idx[k - 1]), int(idx[k + 1])
            d = pos[jn] - pos[jp]
        dn = float(np.hypot(d[0], d[1]))
        heading[i] = math.atan2(float(d[1]), float(d[0])) if dn > 0.05 else (
            float(heading[int(idx[k-1])]) if k > 0 else 0.0
        )

    # Recompute velocity projected onto heading
    vel_out = np.zeros((t_len, 2), dtype=np.float32)
    for i in idx:
        i = int(i)
        spd = math.hypot(float(vel[i, 0]), float(vel[i, 1]))
        h = float(heading[i])
        vel_out[i, 0] = spd * math.cos(h)
        vel_out[i, 1] = spd * math.sin(h)

    tid_str = str(tid)
    tracks[tid_str] = {
        "type": MetaDriveType.VEHICLE,
        "state": {
            "position": pos,
            "velocity": vel_out,
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
scenario = {
    "id": f"highway-merge-in-sumo-{sdc_track_id}",
    "version": "MetaDrive v0.3.0.1",
    "length": t_len,
    "metadata": {
        "metadrive_processed": True,
        "coordinate": MetaDriveType.COORDINATE_METADRIVE,
        "ts": ts,
        "sdc_id": sdc_str,
        "scenario_id": f"hmi_sumo_{sdc_track_id}_{f0}_{f1}",
        "dataset": "highway_merge_in",
        "source_file": os.path.basename(DATASET_DIR),
        "ego_vehicle_class": ego_class,
        "frame_range": (f0, f1),
    },
    "tracks": tracks,
    "dynamic_map_states": {},
    "map_features": map_features,
}
SD.sanity_check(scenario, check_self_type=True)
scenario = SD(scenario)

# ─── Render BEV ───
print(f"\n=== Rendering BEV ===")
os.environ["METADRIVE_HEADLESS"] = "1"
import cv2
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

env = ScenarioOnlineEnv(config=dict(
    use_render=False, image_observation=False,
    agent_policy=ReplayEgoCarPolicy,
    horizon=t_len + 10,
    set_static=True, camera_smooth=False,
    vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False),
))
env.set_scenario(scenario)
env.reset()

# Map-only BEV
from metadrive.utils.draw_top_down_map import draw_top_down_map
map_img = draw_top_down_map(env.current_map, resolution=(2048, 1024), semantic_map=True)
out_map = os.path.join(os.path.dirname(__file__), "docs", "metadrive_bev_sumo.png")
cv2.imwrite(out_map, map_img)
print(f"  Map BEV → {out_map}")

# Mid-frame with vehicles
mid = t_len // 2
for step_i in range(t_len):
    env.step([0.0, 0.0])
    if step_i == mid:
        img = env.render(mode="topdown", window=False, screen_size=(2048, 1024),
                         film_size=(12000, 12000), semantic_map=False, center_on_map=True)
        if img is not None:
            out2 = os.path.join(os.path.dirname(__file__), "docs", "metadrive_bev_sumo_mid.png")
            cv2.imwrite(out2, img)
            print(f"  Mid BEV → {out2}")
        break

env.close()
print("\nDone!")
