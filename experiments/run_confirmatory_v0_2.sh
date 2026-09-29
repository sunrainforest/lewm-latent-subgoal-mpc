#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd -- "$script_dir/.." && pwd)"
source_dir="$repo/source"
python_bin="${PYTHON_BIN:-python}"
output_dir="$repo/runtime/outputs/proprio_v2"

cd "$source_dir"

for seed in $(seq 100 109); do
    baseline_name="confirm_v0_2_baseline_seed${seed}"
    baseline_summary="$output_dir/${baseline_name}_summary.json"
    if [[ -f "$baseline_summary" ]]; then
        echo "SKIP existing $baseline_name"
    else
        echo "START $baseline_name $(date --iso-8601=seconds)"
        "$python_bin" run_proprio_v2_experiment.py \
            --config proprio_v2_config.yaml \
            --stage adapt \
            --freeze-actor \
            --seed "$seed" \
            --episodes 10 \
            --max-steps 300 \
            --run-name "$baseline_name"
        echo "DONE $baseline_name $(date --iso-8601=seconds)"
    fi

    gated_name="confirm_v0_2_gated_h25_seed${seed}"
    gated_summary="$output_dir/${gated_name}_summary.json"
    if [[ -f "$gated_summary" ]]; then
        echo "SKIP existing $gated_name"
    else
        echo "START $gated_name $(date --iso-8601=seconds)"
        "$python_bin" run_proprio_v2_experiment.py \
            --config proprio_v2_config.yaml \
            --stage adapt \
            --oracle-subgoal \
            --oracle-confidence-gate \
            --oracle-lookahead-steps 25 \
            --freeze-actor \
            --seed "$seed" \
            --episodes 10 \
            --max-steps 300 \
            --run-name "$gated_name"
        echo "DONE $gated_name $(date --iso-8601=seconds)"
    fi
done

echo "ALL_DONE $(date --iso-8601=seconds)"
