#!/usr/bin/env python3
"""Compare task-only and play-pretrained planners on the same task episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import run_experiment as base
from latent_subgoal_planner import load_latent_subgoal_planner
from train_latent_subgoal_planner import load_samples, make_dataset
from train_latent_subgoal_planner_stage2 import split_two_way_by_episode


@torch.inference_mode()
def per_sample_mse(model, loader, device) -> np.ndarray:
    result = []
    model.eval()
    for current, goal, phase, proprio, target in loader:
        prediction = model(
            current.to(device), goal.to(device), phase.to(device), proprio.to(device)
        )
        result.append(
            (prediction - target.to(device)).square().mean(dim=-1).cpu().numpy()
        )
    return np.concatenate(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task-only", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = base.load_config(args.config.resolve())
    settings = config["latent_subgoal_planner"]
    task_path = config["paths"]["output_dir"] / "data" / "proprio_v2_task_continuous.npz"
    samples = load_samples(task_path, int(settings["lookahead_steps"]))
    split_seed = int(config["seed"]) + int(settings["split_seed_offset"])
    train_indices, validation_indices, split = split_two_way_by_episode(
        samples["episode"], float(settings["validation_fraction"]), split_seed
    )
    del train_indices
    device = torch.device(args.device)
    loader = DataLoader(make_dataset(samples, validation_indices), batch_size=128)
    task_only, task_payload = load_latent_subgoal_planner(args.task_only, device)
    stage2, stage2_payload = load_latent_subgoal_planner(args.stage2, device)
    expected = sorted(split["validation"])
    stored = sorted(int(value) for value in task_payload["validation_episodes"])
    if expected != stored:
        raise ValueError("Task-only checkpoint validation episodes do not match.")
    if int(task_payload["lookahead_steps"]) != 25 or int(stage2_payload["lookahead_steps"]) != 25:
        raise ValueError("Both checkpoints must use H=25.")

    task_error = per_sample_mse(task_only, loader, device)
    stage2_error = per_sample_mse(stage2, loader, device)
    difference = stage2_error - task_error
    result = {
        "validation_episodes": expected,
        "validation_samples": int(len(validation_indices)),
        "task_only": {
            "latent_mse": float(task_error.mean()),
            "median_sample_mse": float(np.median(task_error)),
        },
        "play_pretrained_then_finetuned": {
            "latent_mse": float(stage2_error.mean()),
            "median_sample_mse": float(np.median(stage2_error)),
        },
        "paired": {
            "stage2_minus_task_only_mean_mse": float(difference.mean()),
            "stage2_better_sample_fraction": float(np.mean(difference < 0)),
            "stage2_equal_sample_fraction": float(np.mean(difference == 0)),
        },
        "interpretation": "Lower latent MSE is better; this is offline prediction, not control success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
