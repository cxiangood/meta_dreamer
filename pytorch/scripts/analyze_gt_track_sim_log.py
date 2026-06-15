#!/usr/bin/env python3
"""Short closed-loop GT PID rollout: log speed/throttle/lateral vs exiD."""
import copy
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from metadrive.envs.scenario_env import ScenarioOnlineEnv
from metadrive.engine.core.engine_core import EngineCore
from metadrive.engine.base_engine import BaseEngine
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.type import MetaDriveType as MT
from metadrive.scenario.scenario_description import ScenarioDescription

from train_online_dreamer_v2 import (
    build_map_features, build_scenario_dict, force_reset_engine,
    BEV_W, BEV_H,
)
from envs.world_model_policy import WorldModelPolicy, setup_wm_policy

REC, TID, LOC = 79, 97, 6
MERGE_IDX = 59
DATA = "/share/home/u23516/data/exiD-dataset-v2.1/data"
MAX_STEPS = 120


def main():
    import pandas as pd
    import torch

    features, off_x, off_y = build_map_features(LOC)
    sd, t_len, (f0, f1) = build_scenario_dict(
        REC, TID, LOC, copy.deepcopy(features), off_x, off_y, DATA)
    tracks = pd.read_csv(os.path.join(DATA, f"{REC:02d}_tracks.csv"), low_memory=False)
    ego = tracks[tracks.trackId == TID].sort_values("frame")
    seg = ego[(ego.frame >= f0) & (ego.frame <= f1)]
    gt_speeds = np.sqrt(seg.xVelocity.values**2 + seg.yVelocity.values**2).astype(np.float64)
    if len(gt_speeds) < t_len:
        gt_speeds = np.pad(gt_speeds, (0, t_len - len(gt_speeds)), mode="edge")
    else:
        gt_speeds = gt_speeds[:t_len]

    device = torch.device("cpu")
    ckpt = "/share/home/u23516/code/meta_dreamer-main/pytorch/logs/df_ph/checkpoint_step14000.pt"
    ac = "/share/home/u23516/code/meta_dreamer-main/pytorch/logs/bc_vc_base/checkpoint_bc_best.pt"
    from train_online_dreamer_v2 import load_models_for_online
    wm, actor, _, _, _, _, _, cfg = load_models_for_online(ckpt, ac, device, explore_std=1.0)
    setup_wm_policy(wm, actor, bev_size=cfg.bev_size, device=device, gt_speeds=gt_speeds)

    scenario = ScenarioDescription(sd)
    env = None
    try:
        force_reset_engine()
        env = ScenarioOnlineEnv(config=dict(
            use_render=False, image_observation=True,
            agent_policy=WorldModelPolicy,
            horizon=t_len + 200, store_map=False, set_static=True,
            camera_smooth=False, decision_repeat=2,
            vehicle_config=dict(no_wheel_friction=False, show_navi_mark=False,
                                image_source="rgb_camera"),
            sensors={"rgb_camera": (RGBCamera, BEV_W, BEV_H)},
            norm_pixel=False, height_scale=0.01,
        ))
        env.config["vehicle_config"].update(
            {"max_engine_force": 3000.0, "max_brake_force": 600.0, "max_speed_km_h": 130.0},
            allow_add_new_key=True,
        )
        sd["metadata"]["location_id"] = LOC
        env.set_scenario(scenario)
        env.reset()
        ego_v = env.agent

        print("step  desired  speed   thr   steer   lat_nav  lane_ok  out_rd")
        for step in range(MAX_STEPS):
            from envs.world_model_policy import _GLOBAL_PREV_ACTION
            act = _GLOBAL_PREV_ACTION.copy()
            obs, _, done, _, info = env.step([0.0, 0.0])
            spd = float(ego_v.speed) if ego_v else 0.0
            lat = float("nan")
            lane_ok = ego_v.lane is not None if ego_v else False
            try:
                if ego_v and ego_v.navigation is not None:
                    lat = float(abs(ego_v.navigation.current_lateral))
            except Exception:
                pass
            ood = bool(info.get("out_of_road", False))
            des = float(gt_speeds[step]) if step < len(gt_speeds) else float("nan")
            print(f"{step:4d}  {des:7.2f}  {spd:6.2f}  {act[1]:+5.2f}  {act[0]:+5.2f}  "
                  f"{lat:7.2f}  {lane_ok!s:5s}  {ood!s:5s}")
            if done:
                print(f"  -> done at step {step}, info keys: out_of_road={info.get('out_of_road')}")
                break
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        force_reset_engine()


if __name__ == "__main__":
    main()
