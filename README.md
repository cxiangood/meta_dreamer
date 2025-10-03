# MetaDrive + DreamerV3 Integration Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个将 **DreamerV3** (世界模型强化学习算法) 与 **MetaDrive** (自动驾驶模拟器) 集成的项目，用于训练车道保持任务。

## 项目概述

- **框架**: DreamerV3 (基于世界模型的强化学习)
- **环境**: MetaDrive 0.4.3 (自动驾驶模拟器)
- **任务**: 车道保持 (Lane Keeping)
- **平台**: WSL2 Ubuntu-24.04, JAX/XLA with GPU

## 主要特性

### 1. 自定义环境封装
- MetaDrive环境的DreamerV3封装 (`embodied/envs/metadrive_lane_keeping.py`)
- 64x64 RGB图像观测
- 车辆状态观测（速度、加速度、转向等）
- 导航信息（到路径的距离、路径完成度等）

### 2. 复杂奖励函数（7个组件）
```python
reward = (
    time_penalty        # -0.02 每步
    + speed_bonus       # speed * 2.0
    + throttle_bonus    # throttle * 1.0 (当油门 > 0)
    + action_consistency # ±0.2/-0.3 (动作平滑性)
    + sharp_steering_penalty # -1.0 * change (急转弯惩罚)
    + acceleration_bonus # 0.5 (持续加速奖励)
    + termination_penalty # -50(碰撞) / -30(出界) / -20(其他)
)
```

### 3. 探索策略
- **探索偏置**: 训练初期10000步内，油门动作偏向加速（linear decay）
- 鼓励agent主动探索环境而不是原地不动

### 4. 导航模块
- **NodeNetworkNavigation**: 生成waypoint标记和导航线
- 显示功能：
  - 🔵 蓝色方块：waypoint标记
  - 🔴 红色标记：目的地
  - 📏 绿色/蓝色线：导航连线

### 5. 环境配置
- **地图**: map=30 (更长的路段)
- **场景**: 1000个程序生成的场景
- **起点**: 固定在最右侧车道 `(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)`
- **终止条件**: 
  - `crash_vehicle_done=True`
  - `crash_object_done=True`
  - `out_of_road_done=True`

### 6. 自动重置机制
- Episode结束后自动重置环境
- 异常处理：窗口关闭时自动创建headless环境
- 清晰的日志标记（每50步记录一次）

### 7. 模型配置
- **模型**: size12m (deter=2048, hidden=256, classes=16)
- **批次**: batch_size=16, batch_length=64
- **报告**: report_length=16
- **训练比率**: train_ratio=32
- **GPU内存**: XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 (4GB GPU)

## 项目结构

```
Prj_worldmoudle/
├── dreamerv3-main/
│   └── dreamerv3-main/
│       ├── dreamerv3/          # DreamerV3核心算法
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── configs.yaml    # ⭐ 主配置文件
│       │   ├── main.py
│       │   └── rssm.py
│       ├── embodied/           # 环境封装
│       │   ├── envs/
│       │   │   └── metadrive_lane_keeping.py  # ⭐ MetaDrive环境封装
│       │   ├── core/
│       │   ├── jax/
│       │   ├── run/
│       │   └── perf/
│       ├── train_metadrive.py         # 训练入口
│       ├── train_dreamerv3_metadrive.sh  # ⭐ 训练启动脚本
│       └── requirements_metadrive.txt
├── metadrive-main/             # MetaDrive模拟器源码
│   └── metadrive/
│       ├── envs/
│       ├── component/
│       ├── engine/
│       └── ...
└── test_navigation.py          # 导航测试脚本
```

## 快速开始

### 1. 环境准备

```bash
# WSL2 Ubuntu-24.04环境
# 创建虚拟环境
python3 -m venv dreamerv3
source dreamerv3/bin/activate

# 安装依赖
cd dreamerv3-main/dreamerv3-main
pip install -r requirements_metadrive.txt

# 安装MetaDrive
cd ../../metadrive-main
pip install -e .
```

### 2. 训练

```bash
cd dreamerv3-main/dreamerv3-main

# 无渲染训练（推荐）
bash train_dreamerv3_metadrive.sh

# 带渲染训练（调试用）
export METADRIVE_RENDER=1
bash train_dreamerv3_metadrive.sh
```

### 3. 测试导航模块

```bash
# 测试导航waypoints是否正确显示
export METADRIVE_RENDER=1
python test_navigation.py
```

## 配置说明

### DreamerV3配置 (`dreamerv3/configs.yaml`)

```yaml
metadrive_lane_keeping:
  task: metadrive_lane_keeping
  run:
    steps: 1e6              # 总训练步数
    envs: 1                 # 环境数量
    eval_envs: 1           # 评估环境数量
    episode_timeout: 300   # Episode超时
  
  jax:
    platform: gpu
    precision: float16
    
  agent:
    size: 12m              # 模型大小
  
  logger:
    filter: '.*'           # 日志过滤
    # 排除高维数组: 不包含 'train/rand/'
```

### MetaDrive环境配置

关键参数（在 `embodied/envs/metadrive_lane_keeping.py`）：

```python
config = dict(
    # 地图配置
    map=30,                          # 30号地图（更长）
    num_scenarios=1000,              # 1000个场景
    
    # 起点配置
    random_spawn_lane_index=False,   # 禁用随机起点
    
    # 导航模块
    vehicle_config=dict(
        navigation_module=NodeNetworkNavigation,
        show_navi_mark=True,
        show_dest_mark=True,
        show_line_to_navi_mark=True,
        show_line_to_dest=True,
    ),
    
    # Agent配置
    agent_configs={
        DEFAULT_AGENT: dict(
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },
    
    # 奖励权重
    success_reward=100.0,
    out_of_road_penalty=10.0,
    crash_vehicle_penalty=20.0,
    driving_reward=10.0,
    speed_reward=3.0,
)
```

## 训练日志示例

```
[Step 50] Speed: 3.21 m/s, Reward: 2.13
[Step 100] Speed: 5.47 m/s, Reward: 8.94
============================================================
[Episode End] Steps: 127, Reward: -33.25
[Episode End] Reason: OUT_OF_ROAD, Speed: 3.94m/s
============================================================
[RESET TRIGGER] Episode ended, resetting...
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
[RESET] Starting new episode (total episodes so far: ~1)...
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
[RESET] ✓ Episode started with map seed 456
```

## 关键实现细节

### 1. Episode循环机制
```python
def step(self, action):
    if action['reset'] or self._done:
        return self._reset()
    
    # 执行动作
    obs, reward, terminated, truncated, info = self._env.step(metadrive_action)
    
    self._done = terminated or truncated
    # 返回带有 is_last=True 的observation
```

### 2. 探索偏置实现
```python
exploration_bias = max(0.0, 1.0 - self._total_steps / 10000.0)
if exploration_bias > 0.01:
    bias_strength = 1.0 * exploration_bias
    throttle_brake = throttle_brake + bias_strength
```

### 3. 导航模块配置
```python
# 顶层 vehicle_config
vehicle_config=dict(
    navigation_module=NodeNetworkNavigation,  # 关键
    show_navi_mark=True,
    show_dest_mark=True,
)

# Agent-specific配置
agent_configs={
    DEFAULT_AGENT: dict(
        spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
    )
}
```

## 已知问题与解决方案

### 1. ~~AssertionError: (32, 1, 1, 17)~~
- **问题**: 数据流长度不足
- **解决**: 减少 `report_length` 到 16

### 2. ~~NotImplementedError: Logger recording high-dimensional arrays~~
- **问题**: 尝试记录高维numpy数组
- **解决**: 从logger.filter中移除 'train/rand/'

### 3. ~~车辆原地不动~~
- **问题**: 随机探索不足
- **解决**: 添加探索偏置机制 + 奖励优化

### 4. ~~环境不自动重置~~
- **问题**: Episode结束后停止
- **解决**: 实现完整的auto-reset逻辑 + 异常处理

### 5. ~~随机起点问题~~
- **问题**: 车辆每次出现在不同车道
- **解决**: `random_spawn_lane_index=False`

### 6. ~~导航标记不显示~~
- **问题**: navigation_module配置不正确
- **解决**: 正确配置 `vehicle_config` 和 `agent_configs`

## 性能优化

### 日志优化
- 从每10步记录改为每50步记录
- 减少console输出，提高训练速度
- 简化episode摘要信息

### 内存优化
- GPU内存限制: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.25`
- 适用于4GB GPU
- 批次大小: 16

### 渲染优化
- 训练时默认关闭渲染（headless）
- 使用 `METADRIVE_RENDER=1` 启用渲染（调试用）
- 只有第一个环境实例渲染

## 路线图

- [x] 基础环境集成
- [x] 奖励函数设计
- [x] 探索策略实现
- [x] 自动重置机制
- [x] 导航模块配置
- [x] 固定起点设置
- [x] 日志优化
- [ ] 训练效果验证
- [ ] 模型性能评估
- [ ] 更复杂的场景测试
- [ ] 多车道切换任务

## 参考资料

### DreamerV3
- 论文: [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- 官方代码: https://github.com/danijar/dreamerv3

### MetaDrive
- 论文: [MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning](https://arxiv.org/abs/2109.12674)
- 官方文档: https://metadrive-simulator.readthedocs.io/
- GitHub: https://github.com/metadriverse/metadrive

## License

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub: [@cxiangood](https://github.com/cxiangood)
- 项目地址: https://github.com/cxiangood/meta_dreamer

## 致谢

- DreamerV3 团队
- MetaDrive 团队
- JAX 和 XLA 团队
