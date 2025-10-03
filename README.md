# Meta Dreamer

Meta Dreamer 是一个基于 DreamerV2/V3 的元学习强化学习框架，用于多任务学习和快速适应新任务。

## 项目简介

本项目结合了元学习（Meta-Learning）和 Dreamer 算法，旨在训练能够快速适应新环境的强化学习智能体。

## 功能特性

- 🚀 基于 DreamerV2/V3 的世界模型
- 🧠 元学习支持，快速适应新任务
- 🎮 支持多种 Gym 环境
- 📊 完整的训练和评估流程
- 🔧 灵活的配置系统

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- Gym / Gymnasium
- NumPy
- 其他依赖见 `requirements.txt`

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/cxiangood/meta_dreamer.git
cd meta_dreamer
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n meta_dreamer python=3.8
conda activate meta_dreamer

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 训练模型

使用默认配置训练模型：

```bash
python train.py --env CartPole-v1 --episodes 1000
```

使用自定义配置文件：

```bash
python train.py --config configs/cartpole.yaml
```

### 评估模型

评估已训练的模型：

```bash
python evaluate.py --checkpoint checkpoints/model_best.pth --env CartPole-v1
```

### 元学习训练

进行元学习训练（多任务）：

```bash
python meta_train.py --config configs/meta_config.yaml --tasks 5
```

## 项目结构

```
meta_dreamer/
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖包
├── setup.py              # 安装配置文件
├── configs/              # 配置文件目录
│   ├── default.yaml      # 默认配置
│   ├── cartpole.yaml     # CartPole 环境配置
│   └── meta_config.yaml  # 元学习配置
├── src/                  # 源代码目录
│   ├── __init__.py
│   ├── models/          # 模型定义
│   │   ├── dreamer.py   # Dreamer 模型
│   │   ├── world_model.py  # 世界模型
│   │   └── meta_learner.py # 元学习器
│   ├── agents/          # 智能体实现
│   │   └── meta_agent.py
│   ├── envs/            # 环境封装
│   │   └── wrappers.py
│   └── utils/           # 工具函数
│       ├── logger.py    # 日志工具
│       └── config.py    # 配置加载
├── scripts/             # 脚本目录
│   ├── train.py        # 训练脚本
│   ├── evaluate.py     # 评估脚本
│   ├── meta_train.py   # 元学习训练脚本
│   └── visualize.py    # 可视化脚本
├── tests/              # 测试文件
├── checkpoints/        # 模型检查点保存目录
├── logs/              # 训练日志目录
└── data/              # 数据目录
```

## 脚本使用说明

### 训练脚本 (train.py)

基础训练脚本，用于在单个环境中训练 Dreamer 模型。

**参数说明：**

```bash
python train.py [OPTIONS]

选项：
  --env TEXT              环境名称 (默认: CartPole-v1)
  --episodes INTEGER      训练回合数 (默认: 1000)
  --batch-size INTEGER    批次大小 (默认: 32)
  --lr FLOAT             学习率 (默认: 0.0003)
  --config PATH          配置文件路径
  --checkpoint PATH      从检查点继续训练
  --seed INTEGER         随机种子 (默认: 0)
  --device TEXT          设备 (cpu/cuda, 默认: cuda)
  --log-dir PATH         日志保存目录 (默认: logs/)
  --save-dir PATH        模型保存目录 (默认: checkpoints/)
```

**示例：**

```bash
# 基础训练
python train.py --env CartPole-v1 --episodes 1000

# 使用GPU和自定义参数
python train.py --env LunarLander-v2 --episodes 5000 --batch-size 64 --device cuda

# 从检查点继续训练
python train.py --checkpoint checkpoints/model_epoch_100.pth

# 使用配置文件
python train.py --config configs/custom_config.yaml
```

### 评估脚本 (evaluate.py)

评估已训练模型的性能。

**参数说明：**

```bash
python evaluate.py [OPTIONS]

选项：
  --checkpoint PATH       模型检查点路径 (必需)
  --env TEXT             环境名称 (默认: CartPole-v1)
  --episodes INTEGER     评估回合数 (默认: 100)
  --render BOOLEAN       是否渲染环境 (默认: False)
  --seed INTEGER         随机种子 (默认: 0)
  --save-video PATH      保存视频路径
```

**示例：**

```bash
# 基础评估
python evaluate.py --checkpoint checkpoints/model_best.pth --env CartPole-v1

# 评估并渲染
python evaluate.py --checkpoint checkpoints/model_best.pth --render True

# 评估并保存视频
python evaluate.py --checkpoint checkpoints/model_best.pth --save-video videos/eval.mp4 --episodes 10
```

### 元学习训练脚本 (meta_train.py)

使用元学习方法在多个任务上训练模型。

**参数说明：**

```bash
python meta_train.py [OPTIONS]

选项：
  --config PATH          元学习配置文件路径 (必需)
  --tasks INTEGER        任务数量 (默认: 5)
  --meta-episodes INTEGER  元训练回合数 (默认: 1000)
  --adapt-steps INTEGER   适应步数 (默认: 10)
  --meta-lr FLOAT        元学习率 (默认: 0.001)
  --device TEXT          设备 (cpu/cuda, 默认: cuda)
```

**示例：**

```bash
# 元学习训练
python meta_train.py --config configs/meta_config.yaml --tasks 10

# 自定义元学习参数
python meta_train.py --config configs/meta_config.yaml --tasks 5 --meta-episodes 2000 --adapt-steps 20
```

### 可视化脚本 (visualize.py)

可视化训练过程和结果。

**参数说明：**

```bash
python visualize.py [OPTIONS]

选项：
  --log-dir PATH         日志目录 (默认: logs/)
  --type TEXT           可视化类型 (rewards/loss/world_model, 默认: rewards)
  --save PATH           保存图片路径
```

**示例：**

```bash
# 可视化训练奖励曲线
python visualize.py --log-dir logs/experiment1 --type rewards

# 可视化损失曲线并保存
python visualize.py --log-dir logs/experiment1 --type loss --save results/loss_curve.png

# 可视化世界模型预测
python visualize.py --log-dir logs/experiment1 --type world_model
```

## 配置文件

配置文件使用 YAML 格式，示例配置：

```yaml
# configs/default.yaml
env:
  name: CartPole-v1
  max_steps: 500

model:
  hidden_size: 256
  latent_size: 32
  num_layers: 3

training:
  episodes: 1000
  batch_size: 32
  learning_rate: 0.0003
  gamma: 0.99

logging:
  log_interval: 10
  save_interval: 100
```

## 常见问题

### Q: 如何添加新环境？

A: 在 `src/envs/` 目录下创建新的环境包装器，或直接使用 Gym 环境名称。

### Q: 训练需要多长时间？

A: 取决于环境复杂度和硬件配置。简单环境（如 CartPole）在 CPU 上几分钟即可完成，复杂环境可能需要数小时。

### Q: 如何调整超参数？

A: 可以通过命令行参数或修改配置文件来调整超参数。建议先使用默认配置，然后逐步调整。

### Q: GPU 训练速度慢？

A: 检查是否正确安装了 CUDA 版本的 PyTorch，使用 `--device cuda` 参数启用 GPU。

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- [DreamerV2](https://github.com/danijar/dreamerv2) - 世界模型灵感来源
- [DreamerV3](https://github.com/danijar/dreamerv3) - 改进的 Dreamer 算法
- [Meta-Learning](https://github.com/cbfinn/maml) - 元学习方法参考

## 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues: [https://github.com/cxiangood/meta_dreamer/issues](https://github.com/cxiangood/meta_dreamer/issues)
- Email: 1504047409@qq.com

## 更新日志

### v0.1.0 (待发布)
- 初始版本发布
- 实现基础 Dreamer 模型
- 添加元学习支持
- 完善文档和示例