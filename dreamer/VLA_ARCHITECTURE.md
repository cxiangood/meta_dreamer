# VLA + World Model Architecture for Autonomous Driving

## 概述 / Overview

本项目将 **SIGLIP 2** 视觉语言模型与 **DreamerV3** 世界模型结合，构建一个类似 **VLA (Vision-Language-Action)** 的架构，用于自动驾驶任务。

This project integrates **SIGLIP 2** vision-language model with **DreamerV3** world model to build a **VLA (Vision-Language-Action)** style architecture for autonomous driving tasks.

## 架构图 / Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VLA + World Model Architecture                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │  Visual Input    │
                              │  (RGB Camera)    │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │   SIGLIP 2   │   │  Proprio.    │   │  (Future)    │
           │   Vision     │   │  Sensors     │   │  Language    │
           │   Encoder    │   │  (speed etc) │   │  Instruction │
           │   (ViT)      │   │              │   │              │
           └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                  │                  │                  │
                  └──────────────────┼──────────────────┘
                                     │
                           ┌─────────▼─────────┐
                           │   Projection &    │
                           │   Fusion Layer    │
                           │   (Adapter)       │
                           └─────────┬─────────┘
                                     │
                           ┌─────────▼─────────┐
                           │      Tokens       │
                           │   (Visual Rep)    │
                           └─────────┬─────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                                 │
                    ▼                                 ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│     RSSM World Model            │     │       Decoder                   │
│  ┌────────────────────────────┐ │     │  ┌────────────────────────────┐ │
│  │   Deterministic State      │ │     │  │   Image Reconstruction    │ │
│  │   (GRU-based dynamics)     │ │     │  └────────────────────────────┘ │
│  └────────────────────────────┘ │     └─────────────────────────────────┘
│  ┌────────────────────────────┐ │
│  │   Stochastic State         │ │
│  │   (Categorical latent)     │ │
│  └────────────────────────────┘ │
└────────────────┬────────────────┘
                 │
    ┌────────────┼────────────┬────────────┐
    │            │            │            │
    ▼            ▼            ▼            ▼
┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────┐
│ Reward │  │Continue│  │ Value  │  │   VLA Policy   │
│  Head  │  │  Head  │  │  Head  │  │     Head       │
└────────┘  └────────┘  └────────┘  │ ┌────────────┐ │
                                    │ │ Action:    │ │
                                    │ │ - Steering │ │
                                    │ │ - Throttle │ │
                                    │ │ - Brake    │ │
                                    │ └────────────┘ │
                                    └────────────────┘
```

## 文件结构 / File Structure

```
dreamer/dreamerv3/
├── agent.py              # 主Agent类 (支持SIGLIP编码器)
├── configs.yaml          # 配置文件 (包含VLA配置)
├── rssm.py              # RSSM世界模型
├── siglip_encoder.py    # SIGLIP 2视觉编码器
├── vla_world_model.py   # VLA架构集成模块
└── main.py              # 主入口
```

## 关键组件 / Key Components

### 1. SIGLIP 2 Vision Encoder (`siglip_encoder.py`)

三种编码器实现:

- **`SiglipVisionEncoder`**: 使用预训练的SIGLIP 2模型 (需要PyTorch)
- **`SiglipEncoderJAX`**: 纯JAX实现的ViT编码器 (无PyTorch依赖)
- **`VLAEncoder`**: 完整VLA编码器 (支持未来的语言条件)

### 2. World Model (RSSM)

DreamerV3的核心世界模型，包含:
- 确定性状态 (GRU)
- 随机状态 (Categorical)
- 支持长期想象 (Imagination)

### 3. VLA Policy Head

结合世界模型状态和视觉特征的策略头:
- 支持连续动作 (转向、油门)
- 支持离散动作
- 可选的视觉特征残差连接

## 使用方法 / Usage

### 安装依赖

```bash
cd dreamer
pip install -r requirements.txt

# 如果使用预训练SIGLIP (推荐)
pip install transformers torch torchvision
```

### 配置选项

在 `configs.yaml` 中选择编码器类型:

```yaml
# 使用预训练SIGLIP 2 (最佳性能)
agent:
  enc:
    typ: siglip
    siglip:
      siglip_path: "../siglip2-so400m-patch16-256"
      output_dim: 1024
      freeze_backbone: True
      aggregation: mean

# 使用纯JAX ViT (无PyTorch依赖)  
agent:
  enc:
    typ: siglip_jax
    siglip_jax:
      output_dim: 1024
      vit_dim: 512
      vit_layers: 6
```

### 训练命令

```bash
# 使用VLA架构训练MetaDrive车道保持
python -m dreamerv3.main --configs vla_metadrive

# 使用纯JAX编码器 (不需要PyTorch)
python -m dreamerv3.main --configs vla_jax_metadrive

# 调试模式
python -m dreamerv3.main --configs vla_debug
```

### Python API

```python
from dreamerv3.agent import Agent
from dreamerv3.siglip_encoder import SiglipVisionEncoder

# 创建带SIGLIP编码器的Agent
config = ...  # 加载配置
config.enc.typ = 'siglip'
config.enc.siglip.siglip_path = '../siglip2-so400m-patch16-256'

agent = Agent(obs_space, act_space, config)
```

## 预训练模型路径 / Pretrained Model Path

SIGLIP 2 模型应放置在:
```
../siglip2-so400m-patch16-256/
├── config.json
├── model.safetensors
├── preprocessor_config.json
└── ...
```

## 配置详解 / Configuration Details

### VLA MetaDrive 配置

```yaml
vla_metadrive:
  task: metadrive_lane_keeping
  env.metadrive.size: [256, 256]  # 匹配SIGLIP输入尺寸
  agent:
    enc:
      typ: siglip
      siglip:
        siglip_path: "../siglip2-so400m-patch16-256"
        output_dim: 1024      # 输出维度
        freeze_backbone: True  # 冻结SIGLIP权重
        aggregation: mean      # 特征聚合方式
        proj_layers: 2         # 投影层数
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `siglip_path` | SIGLIP模型路径 | "" |
| `output_dim` | 输出特征维度 | 1024 |
| `freeze_backbone` | 是否冻结SIGLIP | True |
| `aggregation` | 特征聚合方式 (cls/mean/patches) | mean |
| `proj_layers` | 投影MLP层数 | 2 |
| `image_size` | 输入图像尺寸 | 256 |

## 未来扩展 / Future Extensions

1. **语言条件控制**: 添加语言指令输入，实现指令跟随
2. **DAgger模仿学习**: 集成专家演示的在线聚合
3. **多任务学习**: 统一的VLA架构处理多种驾驶任务
4. **层次化动作**: 支持高级语义动作和低级控制

## 性能考虑 / Performance Considerations

- **GPU内存**: SIGLIP + DreamerV3 需要约 8-12GB 显存
- **训练速度**: 使用冻结SIGLIP时，主要瓶颈在RSSM
- **推理延迟**: SIGLIP编码约 20-30ms，RSSM推理约 5ms

## 参考文献 / References

- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [SIGLIP](https://arxiv.org/abs/2303.15343)
- [RT-2: Vision-Language-Action Models](https://arxiv.org/abs/2307.15818)
