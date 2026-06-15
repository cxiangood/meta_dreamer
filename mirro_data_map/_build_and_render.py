"""
Generate SUMO .net.xml for highway-merge-in with CORRECT geometry:
  - Y = -localX (ramp is ABOVE main road)
  - Traffic flows right-to-left (high localY → low localY)
  - Lane 3 (ramp) starts at localY ≈ 165, merges into Lane 2 at localY ≈ 10
  - 3.75m standard lane width

Then build scenario with localXY and render BEV.
"""
import os, math
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

DATASET_DIR = "/Users/jiojio/Documents/课题组/毕设/mirro_dataset_on_ramp/Highway-merge-in"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)
NET_XML_PATH = os.path.join(OUTPUT_DIR, "highway_merge.net.xml")

LANE_WIDTH_MAIN = 3.75
LANE_WIDTH_RAMP = 3.0

# ─── Coordinate mapping ───
# MetaDrive/SUMO X = localY (along road, traffic: right→left = high→low)
# MetaDrive/SUMO Y = -localX (ramp ABOVE main road)
def to_sumo(localY, localX):
    return localY, -localX

# Lane centers in SUMO Y (= -localX)
lane1_sy = -(-LANE_WIDTH_MAIN / 2)                          # 1.875
lane2_sy = -(-(LANE_WIDTH_MAIN + LANE_WIDTH_MAIN / 2))      # 5.625

# ─── Data analysis ───
print("=== Trajectory data ===")
traj = pd.read_csv(os.path.join(DATASET_DIR, "Trajectory.csv"))
traj.columns = [c.strip() for c in traj.columns]
meta = pd.read_csv(os.path.join(DATASET_DIR, "TrackIDstate.csv"))
for lid in [1, 2, 3]:
    sub = traj[traj["laneId"] == lid]
    print(f"  Lane {lid}: localX median={sub.localX.median():.2f}  "
          f"localY [{sub.localY.min():.1f}, {sub.localY.max():.1f}]")

# ─── Ramp slab centerline from data ───
ramp_data = traj[traj["laneId"] == 3].dropna(subset=["localX", "localY"])
lane2_data = traj[traj["laneId"] == 2].dropna(subset=["localX", "localY"])

slab_step = 10.0
ly_vals = ramp_data["localY"].values
ly_min, ly_max = np.percentile(ly_vals, 1), np.percentile(ly_vals, 99)
edges = np.arange(math.floor(ly_min / slab_step) * slab_step, ly_max + slab_step, slab_step)

ramp_slabs = []
for i in range(len(edges) - 1):
    m = (ramp_data["localY"] >= edges[i]) & (ramp_data["localY"] < edges[i + 1])
    if m.sum() < 5:
        continue
    ramp_slabs.append((0.5 * (edges[i] + edges[i + 1]),
                       float(ramp_data.loc[m, "localX"].median())))

lane2_slabs = {}
for i in range(len(edges) - 1):
    m = (lane2_data["localY"] >= edges[i]) & (lane2_data["localY"] < edges[i + 1])
    if m.sum() < 5:
        continue
    mid = 0.5 * (edges[i] + edges[i + 1])
    lane2_slabs[mid] = float(lane2_data.loc[m, "localX"].median())

ramp_slabs = np.array(ramp_slabs)
print(f"\n=== Ramp geometry ===")
print(f"  Slab points: {len(ramp_slabs)}, localY [{ramp_slabs[-1][0]:.0f}, {ramp_slabs[0][0]:.0f}]")

# Build ramp shape in SUMO coords (X=localY, Y=-localX)
ramp_shape_pts = []
for ly, lx in ramp_slabs[np.argsort(-ramp_slabs[:, 0])]:  # decreasing localY (driving dir)
    sx, sy = to_sumo(ly, lx)
    ramp_shape_pts.append((sx, sy))

# Extend ramp to merge at localY≈10 with convergence curve
last_sx, last_sy = ramp_shape_pts[-1]
merge_end_ly = 10.0
merge_target_lx = lane2_slabs.get(merge_end_ly, -5.625)
merge_sx, merge_sy = to_sumo(merge_end_ly, merge_target_lx)

n_curve = 8
for i in range(1, n_curve + 1):
    t = i / n_curve
    ts = 3 * t**2 - 2 * t**3
    ramp_shape_pts.append((last_sx + ts * (merge_sx - last_sx),
                           last_sy + ts * (merge_sy - last_sy)))

print(f"  Shape: {len(ramp_slabs)} data + {n_curve} curve = {len(ramp_shape_pts)} pts")
print(f"  Start (ramp entrance): SUMO ({ramp_shape_pts[0][0]:.0f}, {ramp_shape_pts[0][1]:.2f})")
print(f"  End   (merge into L2): SUMO ({ramp_shape_pts[-1][0]:.0f}, {ramp_shape_pts[-1][1]:.2f})")

# ─── Generate .net.xml ───
ramp_shape_str = " ".join(f"{sx:.2f},{sy:.2f}" for sx, sy in ramp_shape_pts)

root = ET.Element("net", version="1.9")
ET.SubElement(root, "location", netOffset="0,0",
              convBoundary="-10,0,220,10", origBoundary="-10,0,220,10")

road_x_start, road_x_end = 220.0, -10.0
merge_node_sx, merge_node_sy = to_sumo(merge_end_ly, merge_target_lx)
ramp_start_sx, ramp_start_sy = to_sumo(ramp_slabs[0][0],
                                         ramp_slabs[0][1] + LANE_WIDTH_RAMP / 2)

ET.SubElement(root, "node", id="main_start", x=f"{road_x_start:.2f}", y=f"{lane1_sy:.2f}", type="priority")
ET.SubElement(root, "node", id="main_end", x=f"{road_x_end:.2f}", y=f"{lane1_sy:.2f}", type="priority")
ET.SubElement(root, "node", id="ramp_start", x=f"{ramp_start_sx:.2f}", y=f"{ramp_start_sy:.2f}", type="priority")
ET.SubElement(root, "node", id="merge", x=f"{merge_node_sx:.2f}", y=f"{merge_node_sy:.2f}", type="zipper")

def add_edge(eid, from_n, to_n, n_lanes, width, speed, shape=None):
    attrs = {"id": eid, "from": from_n, "to": to_n, "priority": "3",
             "numLanes": str(n_lanes), "speed": f"{speed:.2f}"}
    if shape:
        attrs["shape"] = shape
    edge = ET.SubElement(root, "edge", **attrs)
    for i in range(n_lanes):
        ET.SubElement(edge, "lane", id=f"{eid}_{i}", index=str(i),
                      speed=f"{speed:.2f}", width=f"{width:.2f}")

add_edge("main_pre", "main_start", "merge", 2, LANE_WIDTH_MAIN, 33.33)
add_edge("main_post", "merge", "main_end", 2, LANE_WIDTH_MAIN, 33.33)
add_edge("ramp", "ramp_start", "merge", 1, LANE_WIDTH_RAMP, 16.67, shape=ramp_shape_str)
ET.SubElement(root, "connection", **{"from": "ramp", "to": "main_post",
              "fromLane": "0", "toLane": "1", "pass": "1"})

tree = ET.ElementTree(root)
ET.indent(tree, space="  ")
tree.write(NET_XML_PATH, encoding="UTF-8", xml_declaration=True)
print(f"\n  .net.xml → {NET_XML_PATH}")

# ─── Build map_features (SUMO coords: X=localY, Y=-localX) ───
from metadrive.type import MetaDriveType

features = {}

# Main road lanes
for lid, sy, label in [("1", lane1_sy, "Lane 1 (inner/bottom)"),
                        ("2", lane2_sy, "Lane 2 (outer/middle)")]:
    pts = np.array([[road_x_start, sy], [road_x_end, sy]], dtype=np.float32)
    features[lid] = {"type": MetaDriveType.LANE_SURFACE_STREET, "polyline": pts}
    print(f"  {label}: Y={sy:.2f}")

# Ramp centerline
features["3"] = {"type": MetaDriveType.LANE_SURFACE_STREET,
                 "polyline": np.array(ramp_shape_pts, dtype=np.float32)}
print(f"  Ramp: {len(ramp_shape_pts)} pts")

# Boundaries
b12_sy = -(LANE_WIDTH_MAIN)  # In SUMO Y: between Lane 1 (1.875) and Lane 2 (5.625) = 3.75
# Wait, boundary between L1 and L2: midpoint of their Y values
b12_y = (lane1_sy + lane2_sy) / 2  # 3.75
features["boundary_1_2"] = {
    "type": MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
    "polyline": np.array([[road_x_start, b12_y], [road_x_end, b12_y]], dtype=np.float32),
}
# Outer edges
features["edge_inner"] = {
    "type": MetaDriveType.BOUNDARY_LINE,
    "polyline": np.array([[road_x_start, lane1_sy - LANE_WIDTH_MAIN/2],
                          [road_x_end, lane1_sy - LANE_WIDTH_MAIN/2]], dtype=np.float32),
}
features["edge_outer"] = {
    "type": MetaDriveType.BOUNDARY_LINE,
    "polyline": np.array([[road_x_start, lane2_sy + LANE_WIDTH_MAIN/2],
                          [road_x_end, lane2_sy + LANE_WIDTH_MAIN/2]], dtype=np.float32),
}
# Ramp outer edge
ramp_edge = np.array([(sx, sy - LANE_WIDTH_RAMP/2) for sx, sy in ramp_shape_pts], dtype=np.float32)
features["edge_ramp"] = {"type": MetaDriveType.BOUNDARY_LINE, "polyline": ramp_edge}
# Boundary between Lane 2 and ramp (dashed)
b23_pts = [(sx, (sy + lane2_sy) / 2) for sx, sy in ramp_shape_pts if sx >= 80]
if len(b23_pts) >= 2:
    features["boundary_2_3"] = {
        "type": MetaDriveType.LINE_BROKEN_SINGLE_WHITE,
        "polyline": np.array(b23_pts, dtype=np.float32),
    }
print(f"  Total features: {len(features)}")

# ─── Build scenario with flipped Y ───
from metadrive.scenario.scenario_description import ScenarioDescription as SD

if meta["RampVehicle"].any():
    sdc_track_id = int(meta.loc[meta["RampVehicle"], "trackId"].iloc[0])
else:
    sdc_track_id = int(meta["trackId"].iloc[0])

ego_row = meta.loc[meta["trackId"] == sdc_track_id].iloc[0]
f0 = int(ego_row["InitialFrame"])
f1 = f0 + int(ego_row["TotalFrame"]) - 1
ego_class = str(ego_row["VehicleClass"])
window = traj[(traj["frameId"] >= f0) & (traj["frameId"] <= f1)].copy()

candidates = [int(t) for t in window["trackId"].unique() if int(t) != sdc_track_id][:48]
selected = [sdc_track_id] + candidates
meta_by_id = meta.set_index("trackId")
t_len = f1 - f0 + 1
ts = np.arange(t_len, dtype=np.float32) * 0.1

tracks = {}
for tid in selected:
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
        # SUMO coords: X = localY, Y = -localX
        pos[fi, 0] = float(r["localY"])
        pos[fi, 1] = -float(r["localX"])
        pos[fi, 2] = 0.0
        vel[fi, 0] = float(r["xVelocity"])
        vel[fi, 1] = float(r["yVelocity"])
        valid[fi] = True
        w, h = float(r["width"]), float(r["height"])
        ref = 4.5 if "truck" not in vclass.lower() else 8.0
        sc = ref / max(w, h, 1.0)
        length[fi] = max(w, h) * sc
        width[fi] = min(w, h) * sc

    if not valid.any():
        continue
    heading = np.zeros(t_len, dtype=np.float32)
    idx = np.flatnonzero(valid)
    for k, i in enumerate(idx):
        i = int(i)
        if k == 0 and idx.size == 1:
            h = 0.0
        elif k == 0:
            d = pos[int(idx[k + 1])] - pos[i]
        elif k == idx.size - 1:
            d = pos[i] - pos[int(idx[k - 1])]
        else:
            d = pos[int(idx[k + 1])] - pos[int(idx[k - 1])]
        dn = float(np.hypot(d[0], d[1]))
        heading[i] = math.atan2(float(d[1]), float(d[0])) if dn > 0.05 else (
            float(heading[int(idx[k - 1])]) if k > 0 else 0.0)

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
        "state": {"position": pos, "velocity": vel_out, "heading": heading,
                  "valid": valid, "length": length, "width": width, "height": height},
        "metadata": {"type": MetaDriveType.VEHICLE, "object_id": tid_str,
                     "dataset": "highway_merge_in"},
    }

sdc_str = str(sdc_track_id)
scenario_dict = {
    "id": f"highway-merge-in-sumo-{sdc_track_id}",
    "version": "MetaDrive v0.3.0.1",
    "length": t_len,
    "metadata": {
        "metadrive_processed": True,
        "coordinate": MetaDriveType.COORDINATE_METADRIVE,
        "ts": ts, "sdc_id": sdc_str,
        "scenario_id": f"hmi_sumo_{sdc_track_id}",
        "dataset": "highway_merge_in",
        "source_file": os.path.basename(DATASET_DIR),
        "ego_vehicle_class": ego_class,
        "frame_range": (f0, f1),
    },
    "tracks": tracks,
    "dynamic_map_states": {},
    "map_features": features,
}
SD.sanity_check(scenario_dict, check_self_type=True)
scenario = SD(scenario_dict)

# ─── Render BEV ───
print(f"\n=== Rendering BEV ===")
os.environ["METADRIVE_HEADLESS"] = "1"
import cv2
from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy

env = ScenarioOnlineEnv(config=dict(
    use_render=False, image_observation=False,
    agent_policy=ReplayEgoCarPolicy,
    horizon=t_len + 10, set_static=True, camera_smooth=False,
    vehicle_config=dict(no_wheel_friction=True, show_navi_mark=False),
))
env.set_scenario(scenario)
env.reset()

from metadrive.utils.draw_top_down_map import draw_top_down_map
map_img = draw_top_down_map(env.current_map, resolution=(2048, 1024), semantic_map=True)
cv2.imwrite(os.path.join(OUTPUT_DIR, "metadrive_bev_sumo.png"), map_img)
print(f"  Map → {OUTPUT_DIR}/metadrive_bev_sumo.png")

mid = t_len // 2
for step_i in range(t_len):
    env.step([0.0, 0.0])
    if step_i == mid:
        img = env.render(mode="topdown", window=False, screen_size=(2048, 1024),
                         film_size=(12000, 12000), semantic_map=False)
        if img is not None:
            cv2.imwrite(os.path.join(OUTPUT_DIR, "metadrive_bev_sumo_mid.png"), img)
            print(f"  Mid → {OUTPUT_DIR}/metadrive_bev_sumo_mid.png")
        break

env.close()
print("Done.")
