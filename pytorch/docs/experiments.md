# Experiments

## Methods

| Abbr | Method |
|------|--------|
| KL | DreamerV3 baseline (KL divergence + decoder) |
| SIG-old | SIGReg on stochastic states only |
| SIG-deter | SIGReg on deterministic state |
| DF+Barlow | Decoder-Free + Barlow Twins (no SIGReg) |
| DF+SIG | Decoder-Free + Barlow + SIGReg (best) |
| DF+SIG+CNN | DF+SIG + CNN learnable downsampling (2x: 300→150) |
| DF+SIG+E2E | DF+SIG with joint WM+AC training (skip Phase 3) |
| DF+SIG+DAPO-P4 | DF+SIG + P4 DAPO: online post-training with K=8, real env + collision proxy |

## Phase 2: World Model Pretraining

| Experiment | Status | WandB | Job ID | Note |
|-----------|--------|-------|--------|------|
| KL | killed (inferior) | `kl_p2_s42` | - | KL=1.00 坍塌 |
| SIG-old | killed (inferior) | `sigreg_old_p2_s42` | - | SIGReg 反向暴涨 |
| SIG-deter | killed (inferior) | `sigreg_deter_continuous_p2_s42` | - | |
| DF+Barlow | killed (inferior) | `df_barlow_p2_s42` | 1440783 | |
| DF+SIG v3 | **running** | `df_sigreg_p2_s42_v3` | 1442910 | step3150, barlow=0.0008 sigreg=0.0036 ✅ |
| DF+SIG+CNN | done → P3 | `df_sig_cnn_p2_s42` | 1441546 | step5600 已收敛，接 CNN P3 v1 (1442912, 已杀) |
| E2E v1 | killed (value infl) | `df_sig_e2e_s42` | 1442851 | value_mean→16.9, λ=0.95 τ=0.02 |
| E2E v2 | **running** | `df_sig_e2e_s42_v2` | 1444484 | step350, value_mean=0.37 ✅ λ=0.5 τ=0.005 |

### DF+SIG 重跑记录 (v1/v2/v3)

- **v1** (5sxzujvz): step 0→2000, barlow=0.005, bs=16, 收敛良好 → ckpt 用于 P3
- **v2** (a8238077/1442847): 续训失败 (barlow loss 上涨 / 改错配置 + Bus error) → 杀
- **v3** (1442910, 当前): 从 P3 ckpt 提取 WM 重建 step2000, barlow=0.005, bs=32, no preload

### DF+SIG+E2E

Jointly train WM + AC in Phase 2, eliminating the need for separate Phase 3 imagination training.

- WM: trained on offline data (same as DF+SIG)
- AC: trained via imagination rollouts from start states extracted from the same batch
- Every WM step also does one AC imagination step (`joint_ac_every=1`)
- After training, can directly run Phase 4 evaluation (no Phase 3 needed)

Command:
```bash
python main.py --phase phase2 --e2e --reg sigreg --sigreg-target deter+logits \
    --use-decoder False --barlow-lambda 0.005 --barlow-k 1 \
    --data-dir /path/to/exid_dreamer_data --logdir logs/df_sig_e2e \
    --bev-size 64 --batch-size 16 --total-steps 500000 --seed 42
```

## Phase 3: Imagination Training

| Experiment | Status | WandB | Job ID | Note |
|-----------|--------|-------|--------|------|
| P3-DF+SIG v1 | killed (value infl) | `df_sig_p3_s42` | 1441661 | value_mean→12, λ=0.95 τ=0.02 |
| P3-DF+SIG v2 | **running** | `df_sig_p3_s42_v2` | 1444599 | 从 P2 v3 ckpt, λ=0.5 τ=0.005 |
| P3-DF+SIG+CNN v1 | killed (value infl) | `df_sig_cnn_p3_s42` | 1442912 | value_mean→10, λ=0.95 τ=0.02 |
| P3-DF+SIG+CNN v2 | **running** | `df_sig_cnn_p3_s42_v2` | 1444483 | step4300 (实际~300), value_mean=0.42 ✅ |
| P4-DAPO v1 | **queued** | `df_sig_dapo_p4_v1` | — | K=8, real env, collision proxy, 从P3 AC ckpt |
| P3-E2E | N/A | — | — | E2E 跳过 Phase 3，直接可跑 P4 |

Phase 3 is only needed for Phase 2 checkpoints that were trained WM-only (without `--e2e`).

## Phase 4: Closed-Loop Evaluation

| Experiment | Status | Checkpoint | Loc | Episodes | Job ID | Result |
|-----------|--------|-----------|-----|----------|--------|--------|
| P4-video | done | df_sigreg_p2 (step2000, random actor) | 1 | 3 | - | — |
| P4-video-v2 | done | df_sigreg_p2 (latest, random actor) | 1 | 3 | - | — |
| P4-eval-P3 v1 | done | df_sig_p3 v1 step4000 AC | 1,3 | 50 | 1442903 | survival 79%, collision 21% |
| P4-online | killed | — | — | — | 1444439 | MetaDrive env 配置问题 |
| P4-eval-CNN-P3 | queued | CNN P3 v2 step6000 | 1,3 | 50 | 1445011 | 排队中 |
| P4-eval-DF-P3 | queued | DF+SIG P3 v2 step4000 | 1,3 | 50 | 1445012 | 排队中 |
| P4-eval-E2E | queued | E2E v2 step2000 | 1,3 | 50 | 1445013 | 排队中 |
| P4-DAPO | queued | DAPO post-train on P3 ckpt | 1,3 | 200 | — | K=8, collision proxy |

### Phase 4 expected results

| Actor source | Expected survival | Notes |
|-------------|-----------------|-------|
| Random (Phase 2 only) | ~0-30% | Episodes end early (arrive_destination), no real decisions |
| Phase 3 trained | 70-90% | Meaningful merge behavior |
| E2E trained | 70-90% | Joint training, skip Phase 3 |
