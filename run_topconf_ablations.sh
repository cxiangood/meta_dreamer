#!/bin/bash
set -euo pipefail

# One-click launcher for NeurIPS-style ablations.
# It submits multiple variants x seeds using submit_il_rl.sh.
# Usage:
#   bash run_topconf_ablations.sh
#   SEEDS="0 1 2 3 4" RUN_STEPS=2e6 bash run_topconf_ablations.sh
#   DRY_RUN=1 bash run_topconf_ablations.sh

ROOT_DIR="/share/home/u23516/code/meta_dreamer-main"
SUBMIT_SCRIPT="$ROOT_DIR/submit_il_rl.sh"

if [[ ! -f "$SUBMIT_SCRIPT" ]]; then
  echo "[ERROR] Missing submit script: $SUBMIT_SCRIPT"
  exit 1
fi

SEEDS=${SEEDS:-"0 1 2 3 4"}
RUN_STEPS=${RUN_STEPS:-"2e6"}
RUN_ENVS=${RUN_ENVS:-"1"}
RUN_EVAL_ENVS=${RUN_EVAL_ENVS:-"1"}
BATCH_SIZE=${BATCH_SIZE:-"16"}
TRAIN_RATIO=${TRAIN_RATIO:-"32"}
DRY_RUN=${DRY_RUN:-"0"}

submit_job() {
  local variant="$1"
  local seed="$2"

  local export_vars="ALL,"
  export_vars+="EXPERIMENT_TAG=${variant},"
  export_vars+="SEED=${seed},"
  export_vars+="RUN_STEPS=${RUN_STEPS},"
  export_vars+="RUN_ENVS=${RUN_ENVS},"
  export_vars+="RUN_EVAL_ENVS=${RUN_EVAL_ENVS},"
  export_vars+="BATCH_SIZE=${BATCH_SIZE},"
  export_vars+="TRAIN_RATIO=${TRAIN_RATIO}"

  case "$variant" in
    baseline)
      export_vars+=",EXPERT_HEADS=1,EXPERT_MODES=1"
      export_vars+=",RISK_THRESHOLD=1.0"
      export_vars+=",ACTION_THRESHOLD=0.0"
      ;;
    risk_only)
      export_vars+=",EXPERT_HEADS=1,EXPERT_MODES=1"
      export_vars+=",RISK_THRESHOLD=0.55"
      export_vars+=",ACTION_THRESHOLD=0.0"
      ;;
    multitraj_only)
      export_vars+=",EXPERT_HEADS=2,EXPERT_MODES=2"
      export_vars+=",RISK_THRESHOLD=1.0"
      export_vars+=",ACTION_THRESHOLD=0.0"
      ;;
    disagreement_only)
      export_vars+=",EXPERT_HEADS=1,EXPERT_MODES=1"
      export_vars+=",RISK_THRESHOLD=1.0"
      export_vars+=",ACTION_THRESHOLD=0.3"
      ;;
    full)
      export_vars+=",EXPERT_HEADS=2,EXPERT_MODES=2"
      export_vars+=",RISK_THRESHOLD=0.55"
      export_vars+=",ACTION_THRESHOLD=0.3"
      ;;
    *)
      echo "[ERROR] Unknown variant: $variant"
      return 1
      ;;
  esac

  local cmd=(sbatch --export="$export_vars" "$SUBMIT_SCRIPT")
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] ${cmd[*]}"
  else
    echo "[SUBMIT] variant=${variant} seed=${seed}"
    "${cmd[@]}"
  fi
}

variants=(baseline risk_only multitraj_only disagreement_only full)

for v in "${variants[@]}"; do
  for s in $SEEDS; do
    submit_job "$v" "$s"
  done
done

echo "[DONE] Submission loop finished."
