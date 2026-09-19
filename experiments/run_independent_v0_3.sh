#!/usr/bin/env bash
set -euo pipefail

repo=/root/yzl/lewm-latent-subgoal-mpc
python_bin=/publicworkspace/le-wm/shared/env/venvs/le-wm-py310/bin/python
source_dir="$repo/source"
output_dir="$repo/runtime/outputs/proprio_v2"
cd "$source_dir"

baseline_name=independent_v0_3_baseline_seed100_199
if [[ ! -f "$output_dir/${baseline_name}_summary.json" ]]; then
    echo "START baseline $(date --iso-8601=seconds)"
    "$python_bin" run_proprio_v2_experiment.py \
        --config proprio_v2_config.yaml --stage adapt --freeze-actor \
        --seed 100 --episodes 100 --max-steps 300 \
        --independent-episode-seeds --run-name "$baseline_name"
    echo "DONE baseline $(date --iso-8601=seconds)"
fi

gated_name=independent_v0_3_gated_h25_seed100_199
if [[ ! -f "$output_dir/${gated_name}_summary.json" ]]; then
    echo "START gated $(date --iso-8601=seconds)"
    "$python_bin" run_proprio_v2_experiment.py \
        --config proprio_v2_config.yaml --stage adapt \
        --oracle-subgoal --oracle-confidence-gate --oracle-lookahead-steps 25 \
        --freeze-actor --seed 100 --episodes 100 --max-steps 300 \
        --independent-episode-seeds --run-name "$gated_name"
    echo "DONE gated $(date --iso-8601=seconds)"
fi

echo "ALL_DONE $(date --iso-8601=seconds)"
