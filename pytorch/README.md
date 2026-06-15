# SIGReg-Dreamer for Highway Merge-in Decision

> 硕士毕设：基于有模型强化学习的自动驾驶汇入决策
> SIGReg-Stabilized DreamerV3 World Model

## 目录结构

```
meta_dreamer_pytorch/
├── main.py                  # 入口 (--phase phase2/3/4 --reg sigreg/kl)
├── config.py                # 超参数 (sigreg_default / kl_default)
├── models/
│   ├── encoder.py           # BEV CNN 编码器+解码器 (3-ch RGB → embed)
│   ├── rssm.py              # RSSM (GRU + categorical, SIGReg/KL 双正则)
│   ├── sigreg.py            # SIGReg (随机投影 + Epps-Pulley 正态性检验)
│   ├── world_model.py       # Encoder+RSSM+Decoder+Reward+Continue
│   └── actor_critic.py      # Actor (tanh-Gaussian) + Critic (two-hot)
├── training/
│   ├── trainer.py           # 三阶段训练 (Phase 2/3/4)
│   ├── offline_buffer.py    # 离线 .npz → 等比例裁剪 → resize → 批次
│   └── replay_buffer.py     # 在线 Replay Buffer (Phase 4)
├── envs/
│   └── metadrive_bev.py     # MetaDrive + RGBCamera BEV 环境
├── utils/
├── slurm_train_sigreg.sh    # A800 SIGReg Phase 2 提交
└── slurm_train_kl.sh        # A800 KL baseline Phase 2 提交
```

## 数据管线

```
exiD dataset → collect_merge_data.py → .npz (400×300 RGB BEV)
                                       ↓
                              offline_buffer.py
                         (中心裁剪 300×300 → resize 64×64)
                                       ↓
                              Phase 2: 世界模型训练
                              Phase 3: 想象策略训练
                              Phase 4: 在线微调 (RGBCamera)
```

## HPC 部署

| 路径 | 说明 |
|------|------|
| `hpcAmd:/share/home/u23516/code/meta_dreamer-main/pytorch/` | 训练代码 |
| `hpcAmd:/share/home/u23516/data/exid_dreamer_data/loc{0,2,4,5,6}/` | 离线数据 |

## 训练命令

```bash
# Phase 2: 离线世界模型 (SIGReg)
python main.py --phase phase2 --reg sigreg \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir logs/sigreg_p2 --bev-size 64 --batch-size 32

# Phase 2: 离线世界模型 (KL baseline)
python main.py --phase phase2 --reg kl \
    --data-dir /share/home/u23516/data/exid_dreamer_data \
    --logdir logs/kl_p2 --bev-size 64 --batch-size 32

# Phase 3: 想象策略训练
python main.py --phase phase3 --reg sigreg \
    --resume logs/sigreg_p2/checkpoint_best.pt \
    --data-dir /share/home/u23516/data/exid_dreamer_data

# Phase 4: 在线微调
python main.py --phase phase4 \
    --resume logs/sigreg_p3/checkpoint_best.pt
```

## SLURM 提交

```bash
cd /share/home/u23516/code/meta_dreamer-main/pytorch
sbatch slurm_train_sigreg.sh   # SIGReg Phase 2 (job 1430326)
sbatch slurm_train_kl.sh       # KL baseline Phase 2 (job 1430327)
```

## 数据格式

`track{T}.npz`:
```
bev_images:  (T, 300, 400, 3) uint8    # RGBCamera BEV 俯视
actions:     (T, 2) float32            # [steering, throttle]
rewards:     (T,) float32
dones:       (T,) bool
positions:   (T, 2) float32            # (可选)
```

## 环境

- **本地**: macOS, meta_dreamer_pytorch/ 开发
- **HPC**: `ssh hpcAmd` → logina.tongji.edu.cn:10022
  - GPU: A800 80GB / L40 24GB
  - Conda: `metadrive` (PyTorch 2.10.0+cu128)
  - 存储: /share (34PB 并行 FS)
