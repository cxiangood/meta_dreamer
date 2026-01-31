# VLA World Model v2 Architecture

## 概述

VLA (Vision-Language-Action) World Model v2 架构将最先进的 VLA 技术与 DreamerV3 世界模型相结合，实现了：

1. **Flow Matching 动作生成** - 使用流匹配替代高斯分布，支持多模态动作分布
2. **Perceiver Resampler** - 高效压缩视觉 token，降低计算成本
3. **Action Chunking** - 预测动作序列以提高时序一致性
4. **语言条件化** - 支持基于语言指令的控制（预留接口）

## 架构图

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    VLA World Model v2                                │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  Image ──→ SIGLIP ViT ──┬──→ Perceiver Resampler ──→ Tokens        │
    │                         │           ↑                               │
    │  Language ──→ LLM ──────┴───────────┘  (可选)                       │
    │                                                                     │
    │  Tokens + Prev Action ──→ RSSM ──→ Latent State                    │
    │                                                                     │
    │  Latent State ──→ Flow Matching Head ──→ Action Chunk              │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. Flow Matching Action Head

Flow Matching 学习一个速度场，将高斯噪声转换为目标动作分布。相比扩散模型：
- 训练更简单（直接回归速度场）
- 推理更快（更直的轨迹，更少的步骤）
- 更适合多模态分布（如十字路口的多种选择）

```python
# 使用 Flow Matching 策略头
policy_head:
  typ: flow
  flow:
    hidden: 512
    layers: 4
    chunk_size: 1        # 动作块大小
    inference_steps: 10  # ODE 积分步数
```

### 2. Perceiver Resampler

将可变长度的视觉 token 压缩为固定数量的 latent tokens：
- 减少计算成本
- 支持多图像/视频输入
- 类似 Flamingo 的设计

### 3. Action Chunking

预测未来 H 步动作，使用指数时序集成：
- 提高时序一致性
- 减少控制抖动
- 类似 ACT/Diffusion Policy 的设计

## 使用方法

### 配置文件

在 `configs.yaml` 中启用 Flow Matching：

```yaml
agent:
  policy_head:
    typ: flow  # 'mlp', 'vla', 或 'flow'
    flow:
      hidden: 512
      layers: 4
      heads: 8
      chunk_size: 1
      inference_steps: 10
      use_transformer: False
      use_visual_residual: True
```

### 训练命令

```bash
python dreamerv3/main.py \
  --config metadrive_on_ramp \
  --agent.policy_head.typ flow \
  --agent.policy_head.flow.hidden 512 \
  --agent.policy_head.flow.layers 4
```

## 损失函数

### Flow Matching Loss

$$
\mathcal{L}_{flow} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t, c) - (x_1 - x_0) \|^2 \right]
$$

其中：
- $x_0 \sim \mathcal{N}(0, I)$ 是噪声
- $x_1$ 是目标动作
- $x_t = (1-t) x_0 + t x_1$ 是插值（OT-CFM 路径）
- $c$ 是条件（世界状态 + 视觉特征）

### ODE 采样

在推理时，通过 Euler 方法积分 ODE：

$$
x_{t+\Delta t} = x_t + v_\theta(x_t, t, c) \cdot \Delta t
$$

从 $t=0$（噪声）到 $t=1$（动作）。

## 与 DreamerV3 的集成

Flow Matching 动作头与 DreamerV3 的 imagination 机制完美集成：

1. **世界模型提供条件** - RSSM 的 latent state 作为 Flow Matching 的条件
2. **想象中使用** - 在 imagination rollout 中采样动作
3. **Actor-Critic 训练** - 仍然使用 TD-λ 回报估计训练

## 文件结构

```
dreamerv3/
├── flow_matching.py      # Flow Matching 核心实现
│   ├── FlowMatchingActionHead
│   ├── PerceiverResampler
│   ├── ActionChunkingPolicy
│   └── LanguageConditioner
├── vla_world_model.py    # VLA + World Model 集成
│   ├── VLAPolicyHead (v1)
│   ├── FlowMatchingPolicyHead (简化版)
│   └── VLAWorldModelV2
└── agent.py              # Agent 集成
```

## 参考文献

1. Flow Matching for Generative Modeling (Lipman et al., 2022)
2. Scaling Rectified Flow Transformers (Esser et al., 2024)
3. π0: A Vision-Language-Action Flow Model (Black et al., 2024)
4. Diffusion Policy (Chi et al., 2023)
5. DreamerV3 (Hafner et al., 2023)
