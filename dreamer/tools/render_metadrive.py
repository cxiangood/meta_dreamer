"""
Render NAVSIM-reconstructed scenario with MetaDrive-style top-down visualization.

Two modes:
  1. --mode pygame: Use pygame for 2D BEV rendering (works headless, no GPU)
  2. --mode engine: Use full MetaDrive engine with 3D rendering (needs GPU/display)

Usage:
    # Pygame top-down rendering (headless)
    python dreamer/tools/render_metadrive.py \
        --scenario logs/map_features/navsim_scenario_test.pkl \
        --output logs/metadrive_render/ --mode pygame --gif

    # Full MetaDrive 3D engine rendering (needs display)
    python dreamer/tools/render_metadrive.py \
        --scenario logs/map_features/navsim_scenario_test.pkl \
        --output logs/metadrive_render/ --mode engine --gif
"""

import argparse
import glob
import os
import pathlib

import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Color palette (MetaDrive-style semantic colors)
# ---------------------------------------------------------------------------
COLORS = {
    "bg": (44, 44, 52),
    "lane_street": (90, 90, 100),
    "lane_unstructure": (75, 75, 85),
    "road_line": (200, 200, 200),
    "road_boundary": (180, 180, 180),
    "ego_fill": (0, 120, 215),
    "ego_outline": (255, 255, 255),
    "ego_trail": (0, 120, 215),
    "vehicle_fill": (230, 150, 50),
    "vehicle_outline": (60, 60, 60),
    "pedestrian": (50, 200, 50),
    "cyclist": (200, 50, 200),
    "cone": (255, 200, 0),
    "barrier": (160, 160, 160),
    "text": (255, 255, 255),
    "heading_arrow": (255, 80, 80),
}


def load_scenario(pkl_path):
    """Load scenario PKL."""
    with open(pkl_path, "rb") as f:
        scenario = pickle.load(f)

    # Convert track states to numpy
    for track_id, track in scenario["tracks"].items():
        state = track["state"]
        for key in ("position", "heading", "velocity", "valid",
                     "length", "width", "height"):
            if key in state:
                state[key] = np.array(state[key], dtype=np.float32)

    # Convert map features to numpy
    for fid, feat in scenario["map_features"].items():
        if "polyline" in feat and feat["polyline"] is not None:
            feat["polyline"] = np.array(feat["polyline"], dtype=np.float64)
        if "polygon" in feat and feat["polygon"] is not None:
            feat["polygon"] = np.array(feat["polygon"], dtype=np.float64)

    return scenario


class BEVRenderer:
    """Bird's Eye View renderer using PIL (no GPU needed)."""

    def __init__(self, scenario, img_size=(1600, 900), view_range=120,
                 follow_ego=True):
        self.scenario = scenario
        self.tracks = scenario["tracks"]
        self.map_features = scenario["map_features"]
        self.length = scenario.get("length", 100)
        self.ego_id = scenario["metadata"].get("sdc_id", "ego")

        self.img_w, self.img_h = img_size
        self.view_range = view_range  # meters in each direction from center
        self.follow_ego = follow_ego

        # Pre-classify map features
        self.lanes = []
        self.connectors = []
        self.boundaries = []
        self._classify_map_features()

    def _classify_map_features(self):
        for fid, feat in self.map_features.items():
            ftype = feat.get("type", "")
            if ftype == "LANE_SURFACE_STREET":
                self.lanes.append((fid, feat))
            elif ftype == "LANE_SURFACE_UNSTRUCTURE":
                self.connectors.append((fid, feat))
            elif "ROAD_LINE" in ftype or "ROAD_EDGE" in ftype:
                self.boundaries.append((fid, feat))

    def _world_to_pixel(self, wx, wy, center_x, center_y):
        """Convert world coords to pixel coords."""
        scale = min(self.img_w, self.img_h) / (2 * self.view_range)
        px = int((wx - center_x) * scale + self.img_w / 2)
        py = int(self.img_h / 2 - (wy - center_y) * scale)  # flip y
        return px, py

    def _draw_polygon(self, draw, polygon, center_x, center_y, fill, outline=None):
        """Draw a polygon in world coords."""
        pts = []
        for p in polygon:
            if len(p) >= 2:
                pts.append(self._world_to_pixel(p[0], p[1], center_x, center_y))
        if len(pts) >= 3:
            draw.polygon(pts, fill=fill, outline=outline)

    def _draw_polyline(self, draw, polyline, center_x, center_y, color, width=1):
        """Draw a polyline in world coords."""
        pts = []
        for p in polyline:
            if len(p) >= 2:
                pts.append(self._world_to_pixel(p[0], p[1], center_x, center_y))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width)

    def _draw_vehicle(self, draw, x, y, heading, length, width, color, outline=None):
        """Draw a rotated rectangle for a vehicle."""
        # Vehicle corners in local frame
        hw, hl = width / 2, length / 2
        corners_local = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]

        # Rotate and translate
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        pts = []
        for cx, cy in corners_local:
            rx = cx * cos_h - cy * sin_h + x
            ry = cx * sin_h + cy * cos_h + y
            pts.append(self._world_to_pixel(rx, ry, *self._center))

        if len(pts) >= 3:
            draw.polygon(pts, fill=color, outline=outline)

        # Draw heading arrow
        arrow_len = length * 0.6
        ax = x + arrow_len * cos_h
        ay = y + arrow_len * sin_h
        p1 = self._world_to_pixel(x, y, *self._center)
        p2 = self._world_to_pixel(ax, ay, *self._center)
        draw.line([p1, p2], fill=COLORS["heading_arrow"], width=2)

    def render_frame(self, frame_idx):
        """Render a single frame as PIL Image."""
        img = Image.new("RGB", (self.img_w, self.img_h), COLORS["bg"])
        draw = ImageDraw.Draw(img)

        # Get ego position for centering
        ego = self.tracks.get(self.ego_id)
        if ego and frame_idx < len(ego["state"]["position"]):
            ego_pos = ego["state"]["position"][frame_idx]
            center_x, center_y = ego_pos[0], ego_pos[1]
        else:
            center_x, center_y = 0, 0

        self._center = (center_x, center_y)

        # Compute scale for culling
        scale = min(self.img_w, self.img_h) / (2 * self.view_range)
        cull_range = self.view_range * 1.5  # draw slightly beyond view

        # 1. Draw lane polygons
        for fid, feat in self.lanes:
            poly = feat.get("polygon")
            if poly is not None and len(poly) >= 3:
                cx = np.mean(poly[:, 0])
                cy = np.mean(poly[:, 1])
                if abs(cx - center_x) > cull_range * 3 or abs(cy - center_y) > cull_range * 3:
                    continue
                self._draw_polygon(draw, poly, center_x, center_y,
                                   COLORS["lane_street"])

        # 2. Draw connector polygons
        for fid, feat in self.connectors:
            poly = feat.get("polygon")
            if poly is not None and len(poly) >= 3:
                cx = np.mean(poly[:, 0])
                cy = np.mean(poly[:, 1])
                if abs(cx - center_x) > cull_range * 3 or abs(cy - center_y) > cull_range * 3:
                    continue
                self._draw_polygon(draw, poly, center_x, center_y,
                                   COLORS["lane_unstructure"])

        # 3. Draw lane centerlines
        for fid, feat in self.lanes:
            pl = feat.get("polyline")
            if pl is not None and len(pl) >= 2:
                cx = np.mean(pl[:, 0])
                cy = np.mean(pl[:, 1])
                if abs(cx - center_x) > cull_range * 3 or abs(cy - center_y) > cull_range * 3:
                    continue
                self._draw_polyline(draw, pl, center_x, center_y,
                                    COLORS["road_line"], width=1)

        # 4. Draw boundaries
        for fid, feat in self.boundaries:
            pl = feat.get("polyline")
            if pl is not None and len(pl) >= 2:
                cx = np.mean(pl[:, 0])
                cy = np.mean(pl[:, 1])
                if abs(cx - center_x) > cull_range * 3 or abs(cy - center_y) > cull_range * 3:
                    continue
                self._draw_polyline(draw, pl, center_x, center_y,
                                    COLORS["road_boundary"], width=1)

        # 5. Draw ego trail (past positions)
        if ego and frame_idx > 0:
            trail_pts = []
            for i in range(max(0, frame_idx - 30), frame_idx + 1):
                pos = ego["state"]["position"][i]
                trail_pts.append(self._world_to_pixel(pos[0], pos[1],
                                                       center_x, center_y))
            if len(trail_pts) >= 2:
                draw.line(trail_pts, fill=COLORS["ego_trail"], width=3)

        # 6. Draw traffic objects
        obj_colors = {
            "VEHICLE": (COLORS["vehicle_fill"], COLORS["vehicle_outline"]),
            "PEDESTRIAN": (COLORS["pedestrian"], None),
            "CYCLIST": (COLORS["cyclist"], None),
            "TRAFFIC_CONE": (COLORS["cone"], None),
            "TRAFFIC_BARRIER": (COLORS["barrier"], None),
        }

        for track_id, track in self.tracks.items():
            if track_id == self.ego_id:
                continue
            state = track["state"]
            if frame_idx >= len(state["valid"]):
                continue
            if not state["valid"][frame_idx]:
                continue

            pos = state["position"][frame_idx]
            dist = np.sqrt((pos[0] - center_x)**2 + (pos[1] - center_y)**2)
            if dist > cull_range:
                continue

            heading = state["heading"][frame_idx]
            length = state["length"][frame_idx, 0] if state["length"].ndim > 1 else state["length"][frame_idx]
            width = state["width"][frame_idx, 0] if state["width"].ndim > 1 else state["width"][frame_idx]

            ttype = track.get("type", "VEHICLE")
            fill, outline = obj_colors.get(ttype, (COLORS["vehicle_fill"], None))

            if ttype in ("TRAFFIC_CONE", "TRAFFIC_BARRIER"):
                # Draw as small dot
                px, py = self._world_to_pixel(pos[0], pos[1], center_x, center_y)
                r = 3
                draw.ellipse([px-r, py-r, px+r, py+r], fill=fill)
            elif ttype == "PEDESTRIAN":
                px, py = self._world_to_pixel(pos[0], pos[1], center_x, center_y)
                r = 4
                draw.ellipse([px-r, py-r, px+r, py+r], fill=fill)
            else:
                self._draw_vehicle(draw, pos[0], pos[1], heading,
                                   length, width, fill, outline)

        # 7. Draw ego vehicle (on top)
        if ego and frame_idx < len(ego["state"]["position"]):
            pos = ego["state"]["position"][frame_idx]
            heading = ego["state"]["heading"][frame_idx]
            length = ego["state"]["length"][frame_idx, 0] if ego["state"]["length"].ndim > 1 else ego["state"]["length"][frame_idx]
            width = ego["state"]["width"][frame_idx, 0] if ego["state"]["width"].ndim > 1 else ego["state"]["width"][frame_idx]

            self._draw_vehicle(draw, pos[0], pos[1], heading,
                               length, width,
                               COLORS["ego_fill"], COLORS["ego_outline"])

        # 8. Draw HUD info
        self._draw_hud(draw, frame_idx, ego, center_x, center_y)

        return img

    def _draw_hud(self, draw, frame_idx, ego, cx, cy):
        """Draw heads-up display info."""
        speed = 0
        if ego and frame_idx < len(ego["state"]["velocity"]):
            vel = ego["state"]["velocity"][frame_idx]
            speed = np.sqrt(vel[0]**2 + vel[1]**2)

        # Semi-transparent HUD bar at top
        hud_h = 40
        draw.rectangle([0, 0, self.img_w, hud_h], fill=(30, 30, 38, 200))

        texts = [
            f"Frame: {frame_idx}/{self.length}",
            f"Speed: {speed:.1f} m/s ({speed*3.6:.0f} km/h)",
            f"Pos: ({cx:.0f}, {cy:.0f})",
            f"Lanes: {len(self.lanes)} | Objects: {len(self.tracks)-1}",
        ]
        x = 15
        for txt in texts:
            draw.text((x, 12), txt, fill=COLORS["text"])
            x += len(txt) * 8 + 30

    def render_all(self, output_dir, step=1, max_frames=None):
        """Render all frames and save as PNGs."""
        out = pathlib.Path(output_dir)
        frames_dir = out / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        n = self.length
        if max_frames:
            n = min(n, max_frames)

        print(f"Rendering {n} frames (step={step})...")
        for i in range(0, n, step):
            img = self.render_frame(i)
            img.save(str(frames_dir / f"frame_{i:04d}.png"))
            if i % 10 == 0:
                print(f"  Frame {i}/{n}")

        # Also render overview (all ego trajectory + full map)
        print("Rendering overview...")
        self._render_overview(out / "overview.png")

        print(f"Frames saved to {frames_dir}")
        return frames_dir

    def _render_overview(self, path):
        """Render a full map overview."""
        # Compute bounding box from ego trajectory
        ego = self.tracks.get(self.ego_id)
        if ego:
            positions = ego["state"]["position"]
            xmin, xmax = positions[:, 0].min(), positions[:, 0].max()
            ymin, ymax = positions[:, 1].min(), positions[:, 1].max()
        else:
            xmin, xmax, ymin, ymax = -500, 500, -500, 500

        margin = 100
        xmin -= margin
        xmax += margin
        ymin -= margin
        ymax += margin

        # Large overview image
        scale_factor = 2.0
        x_range = xmax - xmin
        y_range = ymax - ymin
        img_w = int(x_range * scale_factor)
        img_h = int(y_range * scale_factor)
        img_w = max(img_w, 200)
        img_h = max(img_h, 200)

        img = Image.new("RGB", (img_w, img_h), COLORS["bg"])
        draw = ImageDraw.Draw(img)

        def to_pix(wx, wy):
            px = int((wx - xmin) * scale_factor)
            py = int(img_h - (wy - ymin) * scale_factor)
            return px, py

        # Draw lanes
        for fid, feat in self.lanes:
            poly = feat.get("polygon")
            if poly is not None and len(poly) >= 3:
                pts = [to_pix(p[0], p[1]) for p in poly]
                draw.polygon(pts, fill=COLORS["lane_street"])

        # Draw lane centerlines
        for fid, feat in self.lanes:
            pl = feat.get("polyline")
            if pl is not None and len(pl) >= 2:
                pts = [to_pix(p[0], p[1]) for p in pl]
                draw.line(pts, fill=COLORS["road_line"], width=1)

        # Draw connectors
        for fid, feat in self.connectors:
            pl = feat.get("polyline")
            if pl is not None and len(pl) >= 2:
                pts = [to_pix(p[0], p[1]) for p in pl]
                draw.line(pts, fill=(70, 70, 80), width=1)

        # Draw boundaries
        for fid, feat in self.boundaries:
            pl = feat.get("polyline")
            if pl is not None and len(pl) >= 2:
                pts = [to_pix(p[0], p[1]) for p in pl]
                draw.line(pts, fill=COLORS["road_boundary"], width=1)

        # Draw ego trajectory
        if ego:
            trail_pts = [to_pix(p[0], p[1]) for p in ego["state"]["position"]]
            if len(trail_pts) >= 2:
                draw.line(trail_pts, fill=COLORS["ego_trail"], width=3)

        img.save(str(path))
        print(f"  Overview saved to {path} ({img_w}x{img_h})")


def make_gif(frames_dir, output_path, fps=10, pattern="frame_*.png"):
    """Combine saved images into GIF."""
    files = sorted(glob.glob(str(frames_dir / pattern)))
    if not files:
        print(f"No images found matching {pattern}")
        return

    print(f"Creating GIF from {len(files)} images...")
    images = [Image.open(f) for f in files]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF saved to {output_path} ({len(images)} frames, {fps} fps)")


def render_with_engine(scenario, output_dir, max_steps=50):
    """Render using full MetaDrive engine (needs GPU/display)."""
    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.component.sensors.rgb_camera import RGBCamera

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "use_render": False,
        "image_observation": True,
        "agent_policy": ReplayEgoCarPolicy,
        "horizon": max_steps + 10,
        "show_logo": False,
        "show_fps": False,
        "show_interface": False,
        "vehicle_config": dict(
            show_navi_mark=False,
            image_source="rgb_camera",
        ),
        "sensors": {"rgb_camera": (RGBCamera, 800, 600)},
        "cull_lanes_outside_map": False,
    }

    env = ScenarioOnlineEnv(config=config)
    env.set_scenario(scenario)
    obs, info = env.reset(seed=0)

    # Init top-down renderer
    env.render(mode="topdown", screen_size=(1600, 900), film_size=(9000, 9000))

    for t in range(max_steps):
        import pygame
        bev = env.render(mode="topdown", screen_size=(1600, 900),
                         film_size=(9000, 9000),
                         target_vehicle_heading_up=True, semantic_map=True)
        if bev is not None:
            pygame.image.save(bev, str(out / f"engine_bev_{t:04d}.png"))

        try:
            env.engine.get_sensor("rgb_camera").save_image(
                env.agent, str(out / f"engine_rgb_{t:04d}.jpg"))
        except Exception:
            pass

        obs, reward, terminated, truncated, info = env.step([1, 0.88])
        if terminated or truncated:
            break

    env.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Render NAVSIM scenario")
    parser.add_argument("--scenario", default="logs/map_features/navsim_scenario_test.pkl")
    parser.add_argument("--output", default="logs/metadrive_render/")
    parser.add_argument("--mode", choices=["pygame", "engine"], default="pygame",
                        help="Rendering mode: pygame (2D BEV, headless) or engine (3D, needs GPU)")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--step", type=int, default=1, help="Frame step")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--gif", action="store_true", help="Create GIF animation")
    parser.add_argument("--view_range", type=int, default=80, help="View range in meters")
    args = parser.parse_args()

    print(f"Loading scenario: {args.scenario}")
    scenario = load_scenario(args.scenario)
    print(f"  ID: {scenario.get('id')}")
    print(f"  Length: {scenario.get('length')}")
    print(f"  Tracks: {len(scenario.get('tracks', {}))}")
    print(f"  Map features: {len(scenario.get('map_features', {}))}")

    if args.mode == "pygame":
        renderer = BEVRenderer(scenario, view_range=args.view_range)
        frames_dir = renderer.render_all(args.output, step=args.step,
                                         max_frames=args.max_steps)
        if args.gif:
            make_gif(frames_dir, str(pathlib.Path(args.output) / "bev_animation.gif"),
                     fps=args.fps)

    elif args.mode == "engine":
        out = render_with_engine(scenario, args.output, args.max_steps)
        print(f"Engine render output: {out}")


if __name__ == "__main__":
    main()
