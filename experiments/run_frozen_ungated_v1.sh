#!/usr/bin/env bash
# Reproduce one independently reseeded Baseline / ungated-H25 batch.
set -euo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[0-9]+$ || ! $2 =~ ^[0-9]+$ || $2 -eq 0 ]]; then
    echo "Usage: PYTHON_BIN=/path/to/python bash experiments/run_frozen_ungated_v1.sh START_SEED EPISODES" >&2
    exit 2
fi

start_seed=$1
episodes=$2
last_seed=$((start_seed + episodes - 1))
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python3}
out="$repo/runtime/outputs/proprio_v2"
base="frozen_v1_baseline_seed${start_seed}_${last_seed}"
ours="frozen_v1_ungated_h25_seed${start_seed}_${last_seed}"

for required in \
    "$repo/runtime/model/lewm-cube-mapped/config.json" \
    "$repo/runtime/model/lewm-cube-mapped/weights.pt" \
    "$out/checkpoints/proprio_v2_offline.pt" \
    "$out/data/proprio_v2_task_continuous.npz"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required artifact: $required" >&2
        exit 1
    fi
done

"$python_bin" -c 'import imageio, numpy, ogbench, stable_pretraining, stable_worldmodel, torch, torchvision, yaml; assert torch.cuda.is_available(), "CUDA is required"'

# Never overwrite a previous run, including an interrupted one.
for name in "$base" "$ours"; do
    for path in \
        "$out/${name}_summary.json" \
        "$out/${name}_resolved_config.json" \
        "$out/logs/$name" \
        "$out/videos/$name" \
        "$out/checkpoints/${name}_actor_final.pt"; do
        if [[ -e "$path" ]]; then
            echo "Output already exists; refusing to overwrite: $path" >&2
            exit 1
        fi
    done
done

cd "$repo/source"
echo "Baseline: episode seeds $((start_seed + 11000))-$((last_seed + 11000))"
"$python_bin" run_proprio_v2_experiment.py \
    --config proprio_v2_config.yaml --stage adapt --freeze-actor \
    --seed "$start_seed" --episodes "$episodes" --max-steps 300 \
    --independent-episode-seeds --run-name "$base"

echo "Ungated retrieval H=25: same episode seeds"
"$python_bin" run_proprio_v2_experiment.py \
    --config proprio_v2_config.yaml --stage adapt --freeze-actor \
    --oracle-subgoal --oracle-lookahead-steps 25 \
    --seed "$start_seed" --episodes "$episodes" --max-steps 300 \
    --independent-episode-seeds --run-name "$ours"

"$python_bin" "$repo/experiments/summarize_frozen_ungated_v1.py" \
    --baseline "$out/${base}_summary.json" \
    --ungated "$out/${ours}_summary.json" \
    --start-seed "$start_seed" --episodes "$episodes" \
    --output-json "$out/frozen_v1_seed${start_seed}_${last_seed}_comparison.json" \
    --output-markdown "$out/frozen_v1_seed${start_seed}_${last_seed}_comparison.md"

echo "Done: $out/frozen_v1_seed${start_seed}_${last_seed}_comparison.md"
