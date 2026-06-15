# NAVSIM Scenario Primitive Mining → MetaDrive Reconstruction

## Context

当前项目有丰富的 MetaDrive 环境（on-ramp、lane reduction、lane keeping、DAGGER）和 NAVSIM mini 数据（64 个场景，PKL 格式，含 ego 状态 + 3D 目标跟踪 + 速度）。目标是：从 NAVSIM 挖掘 7 类风险场景原语（scenario primitives），提取可控参数，在 MetaDrive 中重建并做反事实风险放大。

## 新增文件

```
dreamer/tools/
  navsim_miner.py          # Module 1: 解析 PKL + 构建 track + 检测 7 类原语
  primitive_params.py      # Module 2: 提取标准化参数 → ScenarioParameters
  metadrive_builder.py     # Module 3: 从参数构建 MetaDrive 环境 + CutInPolicy
  counterfactual.py        # Module 4: 参数扰动生成风险放大变体
  run_mining_pipeline.py   # CLI 驱动脚本
```

不新建 MetaDrive env wrapper——直接通过 builder 生成可配置的 MetaDrive 环境实例，供现有训练脚本使用。

---

## Phase 1: NAVSIM Scenario Miner (`navsim_miner.py`)

**输入**: `/share/home/u23516/code/navsim_mini/mini_navsim_logs/mini/*.pkl`
**输出**: JSON 文件列表，每个 JSON 描述一个检测到的 scenario primitive

### 关键数据结构

每个 PKL = `List[FrameDict]`，每帧含：
- `ego_dynamic_state`: [vel_x, vel_y, yaw, yaw_rate]
- `ego2global_translation` / `ego2global_rotation`: 4×4 变换
- `anns`: `gt_boxes` [N,7] (ego 系), `gt_names` [N], `gt_velocity_3d` [N,3], `track_tokens` [N]
- `traffic_lights`: [(id, state)]
- `sample_prev` / `sample_next`: 时序链接

### 核心步骤

1. **`load_scenario(pkl_path)`** → `List[FrameDict]`
2. **`build_tracks(frames)`** → 用 `track_tokens` 跨帧关联目标，得到 `Dict[track_token → ObjectTrack]`
   - positions [N,3] (ego 系), positions_global [N,3], velocities [N,3], boxes [N,7]
   - 速度补充：`gt_velocity_3d` 为零时用位置差分 / dt(0.5s)
3. **`detect_primitives(frames, tracks, ego_traj)`** → `List[ScenarioPrimitive]`

### 7 类原语检测规则（基于 ego-centric 坐标，x=前，y=左）

| # | 原语 | 核心判定条件 |
|---|------|-------------|
| 1 | Follow slow vehicle | 前方车辆 (0<x<30m, |y|<4m) 持续 ≥5s，相对速度 < -2 m/s |
| 2 | Lane change w/ rear | ego 横移 >3m（换道）时，后方 (-20<x<0) 有车 |
| 3 | Merge into narrow gap | 前方相邻车道两车间距 <10m，ego 进入该间隙 |
| 4 | Cut-in from adjacent | 前方车辆从 |y|>2m 横移到 |y|<1.5m（切入 ego 车道），0<x<20m |
| 5 | Yield to crossing | 侧方车辆 y 趋近 ego，ego 减速 |
| 6 | Brake for leader | 前车 (0<x<20m, |y|<2m)，TTC<4s，ego 减速 |
| 7 | Blocked lane | ego 车道前方多个静止车辆/锥桶 (0<x<30m, vel<0.5m/s) |

### 输出格式

```json
{
  "primitive_type": "cut_in",
  "source_file": "2021.05.12.22.28.35_veh-35_00620_01164.pkl",
  "start_frame": 234, "end_frame": 278,
  "confidence": 0.85,
  "ego_state": {"speed": 12.3, "acceleration": -1.5, "heading": 0.05},
  "involved_objects": ["track_token_abc123"],
  "object_states": [{"rel_x": 8.5, "rel_y": -1.2, "rel_vx": -3.0, "rel_vy": 0.5}],
  "traffic_density": 0.15,
  "road_type": "straight"
}
```

---

## Phase 2: Parameter Extractor (`primitive_params.py`)

将原始检测结果转为 MetaDrive 可用的标准化参数 `ScenarioParameters`。

```python
@dataclass
class ScenarioParameters:
    map_config: str          # "SSSSS" / "SSrSS" / "SSySS"
    lane_num: int            # 2 or 3
    lane_width: float        # 3.5
    ego_speed: float         # m/s
    ego_spawn_longitude: float
    ego_spawn_lane: int
    traffic_vehicles: List[TrafficVehicleParams]
    primitive_type: str
    duration_seconds: float
    risk_level: float = 0.0

@dataclass
class TrafficVehicleParams:
    spawn_longitude: float
    spawn_lateral: float
    spawn_lane: int
    speed: float
    target_speed: float
    heading: float
    policy: str              # "IDMPolicy" / "CutInPolicy" / "StationaryPolicy"
    policy_kwargs: dict      # {"trigger_time": 2.0, "target_lane_offset": 3.5}
```

### 原语 → MetaDrive 映射

| 原语 | map_config | 交通布局 |
|------|-----------|---------|
| Follow slow | SSSSS, 3车道 | 前方 1 辆 IDMPolicy（低速） |
| Lane change rear | SSSSS, 3车道 | 后方 1 辆 IDMPolicy |
| Narrow gap merge | SSrSS, 3车道 | 前方 2 辆夹出 gap |
| Cut-in | SSSSS, 3车道 | 旁道 1 辆 CutInPolicy |
| Yield crossing | SSrSS | 汇入车辆 IDMPolicy |
| Brake leader | SSSSS, 3车道 | 前方 1 辆 IDMPolicy（低速） |
| Blocked lane | SSySS | 静止障碍物 |

---

## Phase 3: MetaDrive Builder (`metadrive_builder.py`)

从 `ScenarioParameters` 构建可运行的 MetaDrive 环境。

### 核心 API（复用现有模式）

参考 `metadrive_on_ramp.py:166-172` 的 agent 放置：
```python
agent.set_position([x, y], height=agent.HEIGHT/2)
agent.set_heading_theta(theta)
```

参考 `il_then_rl.py` 的交通车辆生成：
```python
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.idm_policy import IDMPolicy
vehicle = env.engine.traffic_manager.spawn_object(DefaultVehicle, ...)
env.engine.traffic_manager.add_policy(vehicle.name, IDMPolicy, control_object=vehicle, ...)
```

### CutInPolicy（内嵌实现）

IDMPolicy 只做纵向控制，cut-in 需要横向运动。实现一个轻量策略：
- 前 N 秒：直行（steering=0），按 target_speed 巡航
- trigger_time 后：施加固定转向使车辆横移到 ego 车道
- 到达目标车道后：恢复直行

---

## Phase 4: Counterfactual Amplifier (`counterfactual.py`)

对基准 `ScenarioParameters` 做参数扰动，生成风险放大变体。

| 扰动轴 | 方法 |
|--------|------|
| 距离 | distance_factor × [0.5, 0.7, 0.85, 1.0] |
| 速度 | speed_factor × [1.0, 1.15, 1.3, 1.5] |
| 间隙 | gap_factor × [1.0, 0.8, 0.6, 0.4] |
| Cut-in 时机 | timing_factor × [1.0, 0.8, 0.6, 0.4] |
| 攻击性 | 混合调节 |

### 每类原语的扰动策略

- **Follow slow / Brake leader**: 减小跟车距离 + 提高 ego 速度
- **Lane change rear**: 后车更近 + 更快
- **Narrow gap**: 缩小间隙 + 提高 ego 速度
- **Cut-in**: 更早切入 + 更快切入速度 + 更近距离
- **Yield**: 提高交叉车辆速度
- **Blocked**: 缩短反应距离 + 提高 ego 速度

---

## CLI 驱动 (`run_mining_pipeline.py`)

```bash
# Step 1: 挖掘原语
python dreamer/tools/run_mining_pipeline.py mine \
  --data_dir /share/home/u23516/code/navsim_mini/mini_navsim_logs/mini/ \
  --output_dir logs/mined_primitives/

# Step 2: 构建参数
python dreamer/tools/run_mining_pipeline.py build \
  --input_dir logs/mined_primitives/ \
  --output_dir logs/scenario_params/

# Step 3: 反事实放大
python dreamer/tools/run_mining_pipeline.py counterfactual \
  --input_dir logs/scenario_params/ \
  --output_dir logs/counterfactual/ \
  --num_variants 8

# 全流程
python dreamer/tools/run_mining_pipeline.py all \
  --data_dir /share/home/u23516/code/navsim_mini/mini_navsim_logs/mini/ \
  --output_dir logs/pipeline_output/
```

---

## 实现顺序

1. **`navsim_miner.py`** — 先实现 load + build_tracks，再实现 7 个检测器（先做简单的 follow slow、brake leader、blocked lane）
2. **`primitive_params.py`** — 原语 → ScenarioParameters 映射
3. **`metadrive_builder.py`** — 环境构建 + CutInPolicy
4. **`counterfactual.py`** — 参数扰动
5. **`run_mining_pipeline.py`** — CLI 串联

## 关键复用文件

- `dreamer/embodied/envs/metadrive_lane_keeping.py` — embodied.Env 接口模式
- `dreamer/embodied/envs/metadrive_on_ramp.py` — agent 放置 + SSrSS 配置
- `dreamer/embodied/envs/metadrive_lane_reduction.py` — SSySS 配置
- `dreamer/embodied/envs/il_then_rl.py` — IDMPolicy 使用 + make_idm_env 工厂
- `dreamer/tools/navsim_instruction_v1.py` — NAVSIM 数据加载模式

## 验证方式

1. Mine: 运行 `mine` 命令，检查输出 JSON 中检测到的原语数量和类型分布是否合理
2. Build: 对每个原语类型，用 builder 生成 MetaDrive 环境，手动跑 1 episode 验证场景行为正确
3. Counterfactual: 对比基准 vs 扰动变体的 TTC、最小距离等指标，确认风险确实被放大
