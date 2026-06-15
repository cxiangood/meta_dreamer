"""
Replay the Chinese Highway-merge-in CSV trajectories in MetaDrive.

Unpack the zip (skip the large PDF if disk is tight)::

    mkdir -p third_party_data/highway_merge_in
    unzip -o ~/Downloads/Highway-merge-in.zip \\
        Highway-merge-in/Trajectory.csv Highway-merge-in/TrackIDstate.csv \\
        Highway-merge-in/Trackstate.txt Highway-merge-in/Road.jpg \\
        -d third_party_data/highway_merge_in

Run::

    python -m metadrive.examples.highway_merge_in_replay --render

Replay uses ``set_static=True`` and disables chase-camera smoothing so physics / smoothing do not fight
frame-wise trajectory playback (removes most visible jitter). Use ``--physics-replay`` to opt out.

**What gets fed into MetaDrive here** is a ``ScenarioDescription`` (Waymo-style replay), not low-level pedals:
each vehicle track has per-frame ``position``, ``heading``, ``velocity`` (2D), plus box size. The
``ReplayEgoCarPolicy`` copies those states into the simulator each step; ``env.step([...])`` actions are
not used to drive the ego. The Highway-merge-in CSV does not ship steering / throttle-brake; those would
require inverse dynamics or a different control policy / env.

Use ``--list-ramp`` to print ramp vehicle track ids, ``--track-id`` to choose ego.
"""
from __future__ import annotations

import argparse
import os
import sys

from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.scenario.highway_merge_in import build_highway_merge_scenario, default_dataset_dir
from metadrive.scenario.scenario_description import ScenarioDescription as SD


def main():
    parser = argparse.ArgumentParser(description="MetaDrive replay for Highway-merge-in CSV data.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Folder containing Trajectory.csv and TrackIDstate.csv (default: repo third_party_data path).",
    )
    parser.add_argument("--track-id", type=int, default=None, help="Ego vehicle trackId (default: first ramp).")
    parser.add_argument("--max-traffic", type=int, default=48, help="Max other vehicles (by proximity).")
    parser.add_argument("--no-ramp-default", action="store_true", help="If no --track-id, use first track in meta.")
    parser.add_argument("--render", action="store_true", help="Open 3D viewer.")
    parser.add_argument("--steps", type=int, default=None, help="Max steps (default: full scenario length).")
    parser.add_argument("--list-ramp", action="store_true", help="Print ramp track ids from TrackIDstate and exit.")
    parser.add_argument(
        "--no-flip-y",
        action="store_true",
        help="Disable default BEV fix (negating posY/yVelocity). Use if the scene appears mirrored.",
    )
    parser.add_argument(
        "--map",
        choices=["slab", "legacy", "clean"],
        default="slab",
        help="Map: slab=5m localY slabs + laneId localX median, affine→pos (说明文档坐标; default); legacy/clean=…",
    )
    parser.add_argument(
        "--clean-map",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--physics-replay",
        action="store_true",
        help="Use Bullet dynamics during replay (default: kinematic/static replay, less jitter).",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir or default_dataset_dir()
    if args.list_ramp:
        import pandas as pd

        meta_path = os.path.join(dataset_dir, "TrackIDstate.csv")
        if not os.path.isfile(meta_path):
            print(f"Missing {meta_path}", file=sys.stderr)
            sys.exit(1)
        m = pd.read_csv(meta_path)
        ramp = m.loc[m["RampVehicle"], "trackId"].tolist()
        print("RampVehicle trackId:", ramp[:40], ("..." if len(ramp) > 40 else ""))
        sys.exit(0)

    if not os.path.isdir(dataset_dir):
        print(
            f"Dataset directory not found: {dataset_dir}\n"
            "Unpack Highway-merge-in.zip (see docstring) or pass --dataset-dir.",
            file=sys.stderr,
        )
        sys.exit(1)

    map_mode = "clean" if getattr(args, "clean_map", False) else args.map
    scenario = build_highway_merge_scenario(
        dataset_dir,
        sdc_track_id=args.track_id,
        max_traffic=args.max_traffic,
        prefer_ramp_ego=not args.no_ramp_default,
        flip_y_axis=not args.no_flip_y,
        clean_map=False,
        map_mode=map_mode,
    )
    assert isinstance(scenario, SD)

    horizon = args.steps or int(scenario["length"]) + 50
    render = args.render
    if render:
        os.environ.pop("METADRIVE_HEADLESS", None)

    use_physics = args.physics_replay
    env = ScenarioOnlineEnv(
        config=dict(
            use_render=render,
            agent_policy=ReplayEgoCarPolicy,
            horizon=horizon,
            set_static=not use_physics,
            # Chase camera averages vehicle pose; replay jumps each frame → keep off to avoid shaking.
            camera_smooth=False,
            # Follow navigation polyline instead of raw chassis when lanes exist → less view wobble.
            use_chase_camera_follow_lane=render,
            vehicle_config=dict(
                no_wheel_friction=True,
                show_navi_mark=False,
            ),
        )
    )
    try:
        env.set_scenario(scenario)
        env.reset()
        n = int(scenario["length"]) if args.steps is None else min(args.steps, int(scenario["length"]))
        for i in range(n):
            o, r, tm, tc, info = env.step([0.0, 0.0])
            if tm or tc:
                break
        print(f"Finished after {i + 1} steps (terminal={tm or tc}).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
