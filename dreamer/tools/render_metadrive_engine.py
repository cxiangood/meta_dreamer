"""
Render NAVSIM-reconstructed scenario using MetaDrive's full 3D engine.

Requires GPU + display/EGL. Designed to run via SLURM with L40/A800.

Usage:
    python dreamer/tools/render_metadrive_engine.py \
        --scenario logs/map_features/navsim_scenario_test.pkl \
        --output logs/metadrive_engine/ \
        --max_steps 100 --fps 12
"""

import argparse
import glob
import os
import pathlib

import numpy as np
import pickle


def load_scenario(pkl_path):
    """Load scenario PKL and fix types."""
    with open(pkl_path, "rb") as f:
        scenario = pickle.load(f)

    for track_id, track in scenario["tracks"].items():
        state = track["state"]
        for key in ("position", "heading", "velocity", "valid",
                     "length", "width", "height"):
            if key in state:
                arr = np.array(state[key], dtype=np.float32)
                # MetaDrive requires 1D arrays for scalar states
                if key in ("heading", "valid") and arr.ndim == 2:
                    arr = arr.squeeze(axis=1)
                if key in ("length", "width", "height") and arr.ndim == 2:
                    arr = arr.squeeze(axis=1)
                state[key] = arr

        # MetaDrive requires metadata with type and object_id
        if "metadata" not in track or not track["metadata"]:
            track["metadata"] = {
                "type": track.get("type", "VEHICLE"),
                "object_id": track_id,
                "track_length": scenario.get("length", 100),
            }

    meta = scenario["metadata"]
    meta.setdefault("sdc_id", "ego")
    meta.setdefault("coordinate", "right-handed")
    meta.setdefault("metadrive_processed", False)
    meta.setdefault("sample_rate", 0.5)
    if "ts" not in meta:
        length = scenario.get("length", 100)
        sr = meta.get("sample_rate", 0.5)
        meta["ts"] = np.arange(length) * sr
    # ts must be numpy array
    meta["ts"] = np.array(meta["ts"], dtype=np.float64)

    # Filter tracks: only keep ego + objects that are ever within 80m of ego
    ego_track = scenario["tracks"].get("ego")
    if ego_track:
        ego_pos = np.array(ego_track["state"]["position"])
        keep = {"ego"}
        for tid, trk in scenario["tracks"].items():
            if tid == "ego":
                continue
            pos = np.array(trk["state"]["position"])
            valid = np.array(trk["state"]["valid"])
            if valid.sum() == 0:
                continue
            # Check min distance to any ego position
            for i in range(len(ego_pos)):
                if valid[i]:
                    d = np.sqrt((pos[i, 0] - ego_pos[i, 0])**2 +
                                (pos[i, 1] - ego_pos[i, 1])**2)
                    if d < 80:
                        keep.add(tid)
                        break
        removed = len(scenario["tracks"]) - len(keep)
        if removed > 0:
            print(f"  Filtered tracks: {len(keep)}/{len(scenario['tracks'])} (removed {removed})")
            scenario["tracks"] = {k: v for k, v in scenario["tracks"].items() if k in keep}

    # dynamic_map_states must be a dict
    if not isinstance(scenario.get("dynamic_map_states"), dict):
        scenario["dynamic_map_states"] = {}

    # object_summary / number_summary required by sanity check
    meta.setdefault("object_summary", {})
    meta.setdefault("number_summary", {})

    return scenario


def render_engine(scenario, output_dir, max_steps=100):
    """Render with full MetaDrive engine."""
    # Patch PBR pipeline for headless GPU if needed
    import metadrive.third_party.simplepbr as _spbr
    from metadrive.third_party.simplepbr import Pipeline as _Pipeline

    _orig_setup = _Pipeline._setup_tonemapping
    def _patched_setup_tonemapping(self):
        try:
            _orig_setup(self)
        except (AttributeError, TypeError):
            self.tonemap_quad = None
            self._shader_ready = False
            print("[PATCH] Tonemapping skipped (headless mode)")
    _Pipeline._setup_tonemapping = _patched_setup_tonemapping

    from metadrive.envs.scenario_env import ScenarioOnlineEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from metadrive.component.sensors.rgb_camera import RGBCamera

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bev").mkdir(exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True)

    config = {
        "use_render": False,
        "image_observation": True,
        "agent_policy": ReplayEgoCarPolicy,
        "no_traffic": False,
        "no_static_vehicles": False,
        "sequential_seed": True,
        "horizon": max_steps + 10,
        "show_logo": False,
        "show_fps": False,
        "show_interface": False,
        "vehicle_config": dict(
            show_navi_mark=False,
            use_special_color=True,
            image_source="rgb_camera",
            lidar=dict(num_lasers=120, distance=50),
            lane_line_detector=dict(num_lasers=0, distance=50),
            side_detector=dict(num_lasers=12, distance=50),
        ),
        "daytime": "12:00",
        "window_size": (800, 450),
        "camera_dist": 9,
        "camera_height": 2.2,
        "camera_fov": 65,
        "sensors": {
            "rgb_camera": (RGBCamera, 800, 600),
        },
        "cull_lanes_outside_map": False,
        "drivable_area_extension": 7,
    }

    print("Creating ScenarioOnlineEnv...")
    env = ScenarioOnlineEnv(config=config)

    print(f"Loading scenario: {scenario.get('id', 'unknown')}")
    print(f"  length={scenario.get('length')}, "
          f"tracks={len(scenario.get('tracks', {}))}, "
          f"map_features={len(scenario.get('map_features', {}))}")

    # Wrap dict in ScenarioDescription
    from metadrive.scenario.scenario_description import ScenarioDescription
    sd = ScenarioDescription(scenario)
    env.set_scenario(sd)

    print("Resetting environment...")
    obs, info = env.reset(seed=0)

    # Initialize top-down renderer
    print("Initializing top-down renderer...")
    env.render(
        mode="topdown",
        screen_size=(1600, 900),
        film_size=(9000, 9000),
        target_vehicle_heading_up=True,
        semantic_map=True,
    )

    print(f"Rendering {max_steps} steps...")
    for t in range(max_steps):
        # Save BEV
        import pygame
        bev_img = env.render(
            mode="topdown",
            screen_size=(1600, 900),
            film_size=(9000, 9000),
            target_vehicle_heading_up=True,
            semantic_map=True,
        )
        if bev_img is not None:
            pygame.image.save(bev_img, str(out / "bev" / f"bev_{t:04d}.png"))

        # Save RGB camera
        try:
            env.engine.get_sensor("rgb_camera").save_image(
                env.agent, str(out / "rgb" / f"rgb_{t:04d}.jpg")
            )
        except Exception as e:
            print(f"  Warning: RGB save failed at step {t}: {e}")

        if t % 10 == 0:
            try:
                agent_pos = env.agent.position
                print(f"  Step {t}/{max_steps}: ego at ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})")
            except Exception:
                print(f"  Step {t}/{max_steps}")

        obs, reward, terminated, truncated, info = env.step([1, 0.88])

        if terminated or truncated:
            print(f"  Episode ended at step {t}")
            break

    print(f"\nDone! Images saved to {out}")
    env.close()
    return out


def make_gif(image_dir, output_path, pattern="*.png", fps=10):
    from PIL import Image as PILImage

    files = sorted(glob.glob(str(image_dir / pattern)))
    if not files:
        print(f"No images found in {image_dir}")
        return

    print(f"Creating GIF from {len(files)} images...")
    images = [PILImage.open(f) for f in files]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"GIF saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="logs/map_features/navsim_scenario_test.pkl")
    parser.add_argument("--output", default="logs/metadrive_engine/")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    out = render_engine(scenario, args.output, args.max_steps)

    make_gif(out / "bev", str(out / "metadrive_bev.gif"), fps=args.fps)
    make_gif(out / "rgb", str(out / "metadrive_rgb.gif"), fps=args.fps)


if __name__ == "__main__":
    main()
