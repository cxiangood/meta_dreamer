"""
渲染所有 7 个 Location 的 BEV 俯视图，验证地图完整性。
"""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from metadrive.utils.sumo.map_utils import RoadLaneJunctionGraph, extract_map_features

MAP_DIR = "/Users/jiojio/metadrive/mirro_data_map"
OUT_DIR = os.path.join(MAP_DIR, "exid_merge_preview")
os.makedirs(OUT_DIR, exist_ok=True)

LOC_MAPS = {
    0: "exid_loc0_orig.net.xml", 1: "exid_loc1_orig.net.xml",
    2: "exid_loc2_orig.net.xml", 3: "exid_loc3_orig.net.xml",
    4: "exid_loc4.net.xml",      5: "exid_loc5_orig.net.xml",
    6: "exid_loc6_orig.net.xml",
}

for loc_id in range(7):
    xml = os.path.join(MAP_DIR, LOC_MAPS[loc_id])
    out_path = os.path.join(OUT_DIR, f"loc{loc_id}_bev.png")
    print(f"\nLocation {loc_id}: {LOC_MAPS[loc_id]}")

    try:
        graph = RoadLaneJunctionGraph(xml)
        features = extract_map_features(graph)
        print(f"  roads={len(graph.roads)} lanes={len(graph.lanes)} features={len(features)}")

        # 收集所有坐标
        all_pts = []
        for k, v in features.items():
            pl = v.get("polyline")
            if pl is not None and len(pl) > 0:
                all_pts.append(np.array(pl)[:, :2])

        if not all_pts:
            print("  ⚠ 无坐标数据")
            continue

        pts = np.vstack(all_pts)
        xmin, ymin = pts.min(axis=0) - 50
        xmax, ymax = pts.max(axis=0) + 50

        # 渲染参数
        H, W = 1024, 2048
        sx = W / (xmax - xmin)
        sy = H / (ymax - ymin)
        scale = min(sx, sy)

        canvas = np.ones((H, W, 3), dtype=np.uint8) * 40  # 深灰背景

        # 分类绘制
        from metadrive.type import MetaDriveType as MT

        for k, v in features.items():
            pl = v.get("polyline")
            if pl is None or len(pl) < 2:
                continue
            pts_arr = np.array(pl)[:, :2]

            # 坐标转换到画布
            screen = ((pts_arr - [xmin, ymin]) * scale).astype(np.int32)
            screen[:, 0] = np.clip(screen[:, 0], 0, W - 1)
            screen[:, 1] = np.clip(screen[:, 1], 0, H - 1)
            # Y 翻转
            screen[:, 1] = H - 1 - screen[:, 1]

            ftype = v.get("type")
            ftype_str = str(ftype)
            if 'SURFACE' in ftype_str:
                color = (80, 80, 80)
                thickness = max(1, int(3.75 * scale * 0.5))
            elif 'BROKEN' in ftype_str:
                color = (180, 180, 180)
                thickness = 1
            elif 'SOLID' in ftype_str:
                color = (200, 200, 200)
                thickness = 2
            else:
                color = (120, 120, 120)
                thickness = 1

            if len(screen) >= 2:
                cv2.polylines(canvas, [screen], False, color, thickness)

        # 标注
        cv2.putText(canvas, f"Location {loc_id} - {LOC_MAPS[loc_id]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f"roads={len(graph.roads)} lanes={len(graph.lanes)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

        cv2.imwrite(out_path, canvas)
        print(f"  ✓ BEV 保存到 {out_path}")

    except Exception as e:
        print(f"  ✗ 错误: {e}")
        import traceback
        traceback.print_exc()

print("\n完成!")
