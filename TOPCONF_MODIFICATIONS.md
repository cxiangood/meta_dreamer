# Top Conference Upgrade Notes (Dreamer + MetaDrive)

This file documents all modifications made to move the project from a pure Dreamer/DAgger baseline toward a stronger, publication-oriented setup.

## 1) Method-Level Changes

### 1.1 Risk-aware expert intervention in DAgger

File:
- dreamer/embodied/envs/metadrive_dagger.py

What was added:
- A risk score that fuses four signals:
  - speed risk
  - lateral-offset risk
  - heading-error risk
  - agent-vs-expert action disagreement risk
- A risk trigger that forces expert intervention when risk score exceeds a threshold.
- New observation keys for analysis and learning:
  - risk_score
  - risk_trigger

Core intent:
- Improve safety and robustness in long-horizon closed-loop control.
- Move from pure schedule-based intervention to state-aware intervention.

### 1.2 Multi-trajectory expert token generation

File:
- dreamer/embodied/envs/metadrive_dagger.py

What was changed:
- expert_heads and expert_modes now produce actual multiple trajectory tokens.
- Token count is expert_heads * expert_modes.
- For each token, a trajectory is generated with slightly perturbed steering/throttle.
- Confidence values are normalized into a probability-like vector.

Core intent:
- Provide richer structured guidance to the world model than a single expert trajectory.
- Enable stronger representation learning and ablation-ready multi-token conditioning.

## 2) Configuration-Level Changes

File:
- dreamer/dreamerv3/configs.yaml

What was added:
- New default MetaDrive risk hyperparameters:
  - risk_threshold
  - risk_speed_weight
  - risk_lateral_weight
  - risk_heading_weight
  - risk_disagreement_weight
  - risk_max_speed
  - risk_max_lateral

What was updated in metadrive_lane_keeping_dagger:
- expert_heads: 2
- expert_modes: 2
- risk-aware settings enabled by default.

Core intent:
- Keep the method configurable and reproducible from a single config source.

## 3) Training Entry Script Upgrades

File:
- submit_il_rl.sh

What was added:
- Reproducibility environment settings:
  - PYTHONHASHSEED
  - CUBLAS_WORKSPACE_CONFIG
  - GIT_COMMIT logging
- Experiment naming for traceability:
  - EXPERIMENT_TAG
  - seed suffix in logdir name
- New runtime-overridable controls:
  - RUN_STEPS, RUN_ENVS, RUN_EVAL_ENVS, BATCH_SIZE, TRAIN_RATIO
  - expert schedule knobs
  - disagreement threshold knob
  - multi-token knobs
  - risk knobs

Core intent:
- Make one script support both final runs and systematic ablations.

## 4) New Ablation Launcher

File:
- run_topconf_ablations.sh

Purpose:
- Submit a full matrix of variants x seeds via Slurm with one command.

Included variants:
- baseline
- risk_only
- multitraj_only
- disagreement_only
- full

Default seeds:
- 0 1 2 3 4

Usage examples:
- bash run_topconf_ablations.sh
- SEEDS="0 1 2 3 4 5 6 7 8 9" RUN_STEPS=2e6 bash run_topconf_ablations.sh
- DRY_RUN=1 bash run_topconf_ablations.sh

## 5) New Statistical Report Tool

File:
- tools/topconf_report.py

Purpose:
- Aggregate run outputs and produce publication-ready statistics.

Input:
- Root directory with run folders containing config.yaml and scores.jsonl.

Output:
- run_details.csv
- summary.csv
- summary.md

Statistics included:
- Mean and std across seeds
- Bootstrap 95% confidence interval
- Permutation-test p-value vs baseline

Usage example:
- python tools/topconf_report.py \
  --logdir /share/home/u23516/code/meta_dreamer-main/dreamer/logs_metadrive \
  --outdir /share/home/u23516/code/meta_dreamer-main/result/topconf_report \
  --tail-episodes 20

## 6) Suggested NeurIPS-style Experiment Protocol

Minimum protocol:
- 5 to 10 seeds per variant
- Same training budget across variants
- Report mean/std/95% CI and p-value
- Include both performance and safety metrics

Recommended primary table:
- Rows: baseline, risk_only, multitraj_only, disagreement_only, full
- Columns:
  - final score (mean +/- std)
  - 95% CI
  - delta vs baseline
  - p-value vs baseline

## 7) Current Limitation and Next Steps

Even with these upgrades, acceptance is never guaranteed. To strengthen submission quality further:
- Add OOD/long-tail scenario evaluation.
- Add failure-case taxonomy with representative trajectories.
- Add compute-cost analysis (extra wall-clock and memory).
- Add a concise theorem/analysis section if possible (risk-trigger stability or intervention bounds).

---

These modifications provide a stronger research baseline, reproducible ablation pipeline, and statistically defensible reporting workflow.
