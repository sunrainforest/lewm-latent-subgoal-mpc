from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_experiment as base
from latent_subgoal_planner import (
    LatentSubgoalPlanner,
    load_latent_subgoal_planner,
)
from train_latent_subgoal_planner import evaluate, load_samples, make_dataset


ALIGN, GRASP, TRANSFER = range(3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain the H=25 latent subgoal planner on OGBench play "
            "trajectories, then fine-tune it on successful task trajectories."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "proprio_v2_config.yaml"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def robot_proprio(observations: np.ndarray) -> np.ndarray:
    closure = np.clip(observations[:, 17:18] / 3.0, 0.0, 1.0)
    return np.concatenate(
        [observations[:, 12:15], observations[:, 15:17], closure], axis=-1
    ).astype(np.float32, copy=False)


def state_ids_from_predicates(
    predicates: np.ndarray,
    proprios: np.ndarray,
    closed_threshold: float,
) -> np.ndarray:
    aligned = predicates[:, 0] >= 0.5
    grasped = (
        (predicates[:, 1] >= 0.5)
        & (proprios[:, 5] >= closed_threshold)
    )
    return np.where(
        grasped, TRANSFER, np.where(aligned, GRASP, ALIGN)
    ).astype(np.int64)


def exact_rows(
    bank_indices: np.ndarray, requested: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.searchsorted(bank_indices, requested)
    valid = rows < len(bank_indices)
    safe_rows = np.minimum(rows, len(bank_indices) - 1)
    valid &= bank_indices[safe_rows] == requested
    return safe_rows, valid


def load_play_samples(
    cache_path: Path,
    source_path: Path,
    lookahead_steps: int,
    closed_threshold: float,
) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        bank_indices = np.asarray(
            cache["unique_source_indices"], dtype=np.int64
        )
        latents = np.asarray(cache["latents"], dtype=np.float32)
        current_rows = np.asarray(cache["current_rows"], dtype=np.int64)
        goal_rows = np.asarray(cache["goal_rows"], dtype=np.int64)
        source_indices = np.asarray(cache["source_indices"], dtype=np.int64)
        goal_indices = np.asarray(cache["goal_indices"], dtype=np.int64)
        episode_ids = np.asarray(cache["episode_ids"], dtype=np.int64)
        predicates = np.asarray(
            cache["predicate_targets"][:, :2], dtype=np.float32
        )
    with np.load(source_path, allow_pickle=False) as source:
        observations = np.asarray(source["observations"], dtype=np.float32)
        terminals = np.asarray(source["terminals"], dtype=bool)

    if not np.all(np.diff(bank_indices) > 0):
        raise ValueError("Latent bank indices must be sorted and unique.")
    if not np.array_equal(bank_indices[current_rows], source_indices):
        raise ValueError("Current latent rows do not match source indices.")
    if not np.array_equal(bank_indices[goal_rows], goal_indices):
        raise ValueError("Goal latent rows do not match goal indices.")

    proprio = robot_proprio(observations[source_indices])
    state_ids = state_ids_from_predicates(
        predicates, proprio, closed_threshold
    )
    terminal_indices = np.flatnonzero(terminals)
    current_episode = np.searchsorted(terminal_indices, source_indices)
    goal_episode = np.searchsorted(terminal_indices, goal_indices)
    if not np.array_equal(current_episode, episode_ids):
        raise ValueError("Episode ids disagree with source terminal boundaries.")

    future_indices = source_indices + lookahead_steps
    target_indices = np.minimum(future_indices, goal_indices)
    target_rows, target_is_encoded = exact_rows(bank_indices, target_indices)
    selected = (
        (state_ids == TRANSFER)
        & (goal_episode == episode_ids)
        & (goal_indices > source_indices)
        & target_is_encoded
    )
    rows = np.flatnonzero(selected)
    if not len(rows):
        raise ValueError("No valid play pretraining samples were found.")
    actual_lookahead = target_indices[rows] - source_indices[rows]
    if np.any(actual_lookahead <= 0) or np.any(
        actual_lookahead > lookahead_steps
    ):
        raise ValueError("Invalid capped lookahead in play samples.")
    return {
        "current": latents[current_rows[rows]].astype(np.float32),
        "goal": latents[goal_rows[rows]].astype(np.float32),
        "phase": state_ids[rows].astype(np.int64),
        "proprio": proprio[rows].astype(np.float32),
        "target": latents[target_rows[rows]].astype(np.float32),
        "episode": episode_ids[rows].astype(np.int64),
        "actual_lookahead": actual_lookahead.astype(np.int64),
        "source_index": source_indices[rows].astype(np.int64),
        "goal_index": goal_indices[rows].astype(np.int64),
        "target_index": target_indices[rows].astype(np.int64),
    }


def split_three_way_by_episode(
    episode_ids: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[int]]]:
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    test_count = max(1, int(round(len(episodes) * test_fraction)))
    if validation_count + test_count >= len(episodes):
        raise ValueError("Validation/test fractions leave no training episodes.")
    validation_episodes = episodes[:validation_count]
    test_episodes = episodes[validation_count : validation_count + test_count]
    train_episodes = episodes[validation_count + test_count :]
    train = np.flatnonzero(np.isin(episode_ids, train_episodes))
    validation = np.flatnonzero(np.isin(episode_ids, validation_episodes))
    test = np.flatnonzero(np.isin(episode_ids, test_episodes))
    return train, validation, test, {
        "train": [int(value) for value in train_episodes],
        "validation": [int(value) for value in validation_episodes],
        "test": [int(value) for value in test_episodes],
    }


def split_two_way_by_episode(
    episode_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    episodes = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    validation_count = max(1, int(round(len(episodes) * validation_fraction)))
    validation_episodes = episodes[:validation_count]
    train_episodes = episodes[validation_count:]
    train = np.flatnonzero(np.isin(episode_ids, train_episodes))
    validation = np.flatnonzero(np.isin(episode_ids, validation_episodes))
    return train, validation, {
        "train": [int(value) for value in train_episodes],
        "validation": [int(value) for value in validation_episodes],
    }


def loader(
    samples: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        make_dataset(samples, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def train_stage(
    model: LatentSubgoalPlanner,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    patience_limit: int,
    stage_name: str,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for current, goal, phase, proprio, target in train_loader:
            current = current.to(device)
            goal = goal.to(device)
            phase = phase.to(device)
            proprio = proprio.to(device)
            target = target.to(device)
            prediction = model(current, goal, phase, proprio)
            loss = nn.functional.mse_loss(
                (prediction - model.latent_mean) / model.latent_std,
                (target - model.latent_mean) / model.latent_std,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        validation = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_normalized_mse": float(np.mean(losses)),
            **validation,
        }
        history.append(record)
        if validation["normalized_mse"] < best_loss:
            best_loss = validation["normalized_mse"]
            best_epoch = epoch
            best_state = clone_state(model)
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps({"stage": stage_name, **record}))
        if patience >= patience_limit:
            break
    if best_state is None:
        raise RuntimeError(f"{stage_name} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_normalized_mse": best_loss,
        "epochs_run": len(history),
        "history": history,
        "model_state": best_state,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(base.json_value(value), output, indent=2, ensure_ascii=False)
        output.write("\n")


def sample_summary(
    samples: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, Any]:
    lookahead = samples["actual_lookahead"][indices]
    return {
        "samples": int(len(indices)),
        "episodes": int(len(np.unique(samples["episode"][indices]))),
        "actual_lookahead_mean": float(np.mean(lookahead)),
        "full_h25_fraction": float(np.mean(lookahead == 25)),
    }


def main() -> None:
    args = parse_args()
    config = base.load_config(args.config.resolve())
    settings = config["latent_subgoal_planner"]
    stage2 = settings.get("stage2", {})
    seed = int(config["seed"] if args.seed is None else args.seed)
    device = torch.device(config["device"] if args.device is None else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    set_seed(seed)

    lookahead_steps = int(settings["lookahead_steps"])
    if lookahead_steps != 25:
        raise ValueError("Stage-two experiment is frozen to H=25.")
    play_samples = load_play_samples(
        config["paths"]["source_latent_cache"],
        config["paths"]["action_dataset"],
        lookahead_steps,
        float(config["labels"]["closed_gripper_threshold"]),
    )
    task_path = (
        config["paths"]["output_dir"]
        / "data"
        / "proprio_v2_task_continuous.npz"
    )
    task_samples = load_samples(task_path, lookahead_steps)

    split_seed = seed + int(settings["split_seed_offset"])
    play_train, play_validation, play_test, play_episodes = (
        split_three_way_by_episode(
            play_samples["episode"],
            float(stage2.get("play_validation_fraction", 0.10)),
            float(stage2.get("play_test_fraction", 0.10)),
            split_seed,
        )
    )
    task_train, task_validation, task_episodes = split_two_way_by_episode(
        task_samples["episode"],
        float(settings["validation_fraction"]),
        # Match the original task-only trainer exactly so its frozen checkpoint
        # can be evaluated on the same held-out task episodes.
        split_seed,
    )

    batch_size = int(stage2.get("batch_size", settings["batch_size"]))
    play_train_loader = loader(
        play_samples, play_train, batch_size, True, seed
    )
    play_validation_loader = loader(
        play_samples, play_validation, batch_size, False, seed
    )
    play_test_loader = loader(
        play_samples, play_test, batch_size, False, seed
    )
    task_train_loader = loader(
        task_samples, task_train, batch_size, True, seed + 1
    )
    task_validation_loader = loader(
        task_samples, task_validation, batch_size, False, seed
    )

    # Normalization is learned only from play-training episodes. Validation,
    # play-test, and all 64-task trajectories remain outside this calculation.
    current = torch.from_numpy(play_samples["current"][play_train])
    goal = torch.from_numpy(play_samples["goal"][play_train])
    target = torch.from_numpy(play_samples["target"][play_train])
    latent_values = torch.cat([current, goal, target], dim=0)
    proprio = torch.from_numpy(play_samples["proprio"][play_train])

    model_config = {
        "latent_dim": int(config["model"]["latent_dim"]),
        "proprio_dim": int(config["model"]["proprio_dim"]),
        "state_count": int(config["model"]["state_count"]),
        "hidden_dim": int(settings["hidden_dim"]),
        "phase_embedding_dim": int(settings["phase_embedding_dim"]),
    }
    model = LatentSubgoalPlanner(**model_config).to(device)
    model.set_normalization(
        latent_values.mean(dim=0).to(device),
        latent_values.std(dim=0).clamp_min(1e-4).to(device),
        proprio.mean(dim=0).to(device),
        proprio.std(dim=0).clamp_min(1e-4).to(device),
    )

    output_dir = config["paths"]["output_dir"]
    pretrain_checkpoint = output_dir / "checkpoints" / stage2.get(
        "pretrain_checkpoint_name",
        "latent_subgoal_planner_play_h25_pretrain_only.pt",
    )
    final_checkpoint = output_dir / "checkpoints" / stage2.get(
        "checkpoint_name", "latent_subgoal_planner_play_finetuned_h25.pt"
    )
    summary_path = output_dir / stage2.get(
        "summary_name", "latent_subgoal_planner_stage2_h25_summary.json"
    )
    for path in (pretrain_checkpoint, final_checkpoint, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}; pass --overwrite explicitly."
            )
    pretrain_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    pretrain_epochs = int(
        stage2.get("pretrain_epochs", 150)
        if args.pretrain_epochs is None
        else args.pretrain_epochs
    )
    finetune_epochs = int(
        stage2.get("finetune_epochs", 200)
        if args.finetune_epochs is None
        else args.finetune_epochs
    )
    common = {
        "weight_decay": float(settings["weight_decay"]),
        "max_grad_norm": float(settings["max_grad_norm"]),
        "patience_limit": int(stage2.get("early_stopping_patience", 30)),
    }
    pretrain_result = train_stage(
        model,
        play_train_loader,
        play_validation_loader,
        device,
        epochs=pretrain_epochs,
        learning_rate=float(stage2.get("pretrain_learning_rate", 2e-4)),
        stage_name="play_pretraining",
        **common,
    )
    pretrain_metrics = {
        "play_train": evaluate(model, loader(play_samples, play_train, batch_size, False, seed), device),
        "play_validation": evaluate(model, play_validation_loader, device),
        "play_test": evaluate(model, play_test_loader, device),
        "task_validation_zero_shot": evaluate(model, task_validation_loader, device),
    }
    torch.save(
        {
            "model_config": model.model_config(),
            "model_state": pretrain_result["model_state"],
            "lookahead_steps": lookahead_steps,
            "phase": "transfer",
            "seed": seed,
            "training_stage": "play_pretraining_only",
            "metrics": pretrain_metrics,
        },
        pretrain_checkpoint,
    )

    finetune_result = train_stage(
        model,
        task_train_loader,
        task_validation_loader,
        device,
        epochs=finetune_epochs,
        learning_rate=float(stage2.get("finetune_learning_rate", 5e-5)),
        stage_name="task_finetuning",
        **common,
    )
    final_metrics = {
        "task_train": evaluate(model, loader(task_samples, task_train, batch_size, False, seed), device),
        "task_validation": evaluate(model, task_validation_loader, device),
        "play_test_after_finetuning": evaluate(model, play_test_loader, device),
    }
    task_only_checkpoint = output_dir / "checkpoints" / settings.get(
        "checkpoint_name", "latent_subgoal_planner_h25.pt"
    )
    if not task_only_checkpoint.exists():
        release_candidate = (
            HERE.parent
            / "release_assets"
            / settings.get("checkpoint_name", "latent_subgoal_planner_h25.pt")
        )
        if release_candidate.exists():
            task_only_checkpoint = release_candidate
    task_only_metrics = None
    if task_only_checkpoint.exists():
        task_only_model, task_only_payload = load_latent_subgoal_planner(
            task_only_checkpoint, device
        )
        if int(task_only_payload["lookahead_steps"]) != lookahead_steps:
            raise ValueError("Task-only baseline checkpoint uses another horizon.")
        task_only_metrics = {
            "checkpoint": str(task_only_checkpoint),
            "task_train": evaluate(
                task_only_model,
                loader(task_samples, task_train, batch_size, False, seed),
                device,
            ),
            "task_validation": evaluate(
                task_only_model, task_validation_loader, device
            ),
        }
    final_payload = {
        "model_config": model.model_config(),
        "model_state": finetune_result["model_state"],
        "lookahead_steps": lookahead_steps,
        "phase": "transfer",
        "seed": seed,
        "training_stage": "play_pretraining_then_task_finetuning",
        "pretrain_checkpoint": str(pretrain_checkpoint),
        "pretrain_metrics": pretrain_metrics,
        "metrics": final_metrics,
    }
    torch.save(final_payload, final_checkpoint)

    summary = {
        "design": {
            "lookahead_steps": lookahead_steps,
            "play_labels": "same-episode observed future latent, capped at sampled goal",
            "play_success_filter": False,
            "play_phase_filter": "proprio_v2 transfer",
            "normalization_source": "play training episodes only",
            "episode_level_splits": True,
            "previous_control_evaluation_seeds_used_for_training": False,
        },
        "inputs": {
            "play_cache_sha256": sha256(config["paths"]["source_latent_cache"]),
            "play_source_sha256": sha256(config["paths"]["action_dataset"]),
            "task_trajectory_sha256": sha256(task_path),
        },
        "samples": {
            "play_total": sample_summary(play_samples, np.arange(len(play_samples["current"]))),
            "play_train": sample_summary(play_samples, play_train),
            "play_validation": sample_summary(play_samples, play_validation),
            "play_test": sample_summary(play_samples, play_test),
            "task_total": sample_summary(task_samples, np.arange(len(task_samples["current"]))),
            "task_train": sample_summary(task_samples, task_train),
            "task_validation": sample_summary(task_samples, task_validation),
        },
        "episode_splits": {
            "play": play_episodes,
            "task": task_episodes,
        },
        "pretraining": {
            **{key: value for key, value in pretrain_result.items() if key != "model_state"},
            "metrics": pretrain_metrics,
            "checkpoint": str(pretrain_checkpoint),
        },
        "finetuning": {
            **{key: value for key, value in finetune_result.items() if key != "model_state"},
            "metrics": final_metrics,
            "checkpoint": str(final_checkpoint),
        },
        "matched_task_only_baseline": task_only_metrics,
    }
    write_json(summary_path, summary)
    print(
        json.dumps(
            base.json_value(
                {
                    "pretrain_checkpoint": pretrain_checkpoint,
                    "final_checkpoint": final_checkpoint,
                    "summary": summary_path,
                    "pretrain_metrics": pretrain_metrics,
                    "final_metrics": final_metrics,
                    "matched_task_only_baseline": task_only_metrics,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
