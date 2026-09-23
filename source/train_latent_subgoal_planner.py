from __future__ import annotations

import argparse
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
from latent_subgoal_planner import LatentSubgoalPlanner


TRANSFER = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the fixed-lookahead latent subgoal planner."
    )
    parser.add_argument(
        "--config", type=Path, default=HERE / "proprio_v2_config.yaml"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--lookahead-steps", type=int, default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--summary-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def future_rows(
    episode_ids: np.ndarray,
    lookahead_steps: int,
) -> np.ndarray:
    result = np.empty(len(episode_ids), dtype=np.int64)
    for episode in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode)
        positions = np.minimum(
            np.arange(len(rows), dtype=np.int64) + lookahead_steps,
            len(rows) - 1,
        )
        result[rows] = rows[positions]
    return result


def load_samples(
    path: Path,
    lookahead_steps: int,
) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Task trajectory dataset missing: {path}. Run v2 training first."
        )
    with np.load(path) as dataset:
        values = {
            name: np.asarray(dataset[name])
            for name in (
                "latents",
                "goal_latents",
                "proprios",
                "state_ids",
                "episode_ids",
                "successes",
            )
        }
    target_rows = future_rows(values["episode_ids"], lookahead_steps)
    selected = (
        (values["successes"] > 0)
        & (values["state_ids"] == TRANSFER)
        & (target_rows > np.arange(len(target_rows)))
    )
    rows = np.flatnonzero(selected)
    return {
        "current": values["latents"][rows].astype(np.float32),
        "goal": values["goal_latents"][rows].astype(np.float32),
        "phase": values["state_ids"][rows].astype(np.int64),
        "proprio": values["proprios"][rows].astype(np.float32),
        "target": values["latents"][target_rows[rows]].astype(np.float32),
        "episode": values["episode_ids"][rows].astype(np.int64),
        "actual_lookahead": (
            target_rows[rows] - rows
        ).astype(np.int64),
    }


def split_by_episode(
    samples: dict[str, np.ndarray],
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    episodes = np.unique(samples["episode"])
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    validation_count = max(
        1, int(round(len(episodes) * validation_fraction))
    )
    validation_episodes = episodes[:validation_count]
    validation = np.isin(samples["episode"], validation_episodes)
    return (
        np.flatnonzero(~validation),
        np.flatnonzero(validation),
        [int(value) for value in validation_episodes],
    )


def make_dataset(
    samples: dict[str, np.ndarray], indices: np.ndarray
) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(samples["current"][indices]),
        torch.from_numpy(samples["goal"][indices]),
        torch.from_numpy(samples["phase"][indices]),
        torch.from_numpy(samples["proprio"][indices]),
        torch.from_numpy(samples["target"][indices]),
    )


@torch.inference_mode()
def evaluate(
    model: LatentSubgoalPlanner,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    squared_error = 0.0
    normalized_error = 0.0
    current_error = 0.0
    goal_error = 0.0
    elements = 0
    samples = 0
    for current, goal, phase, proprio, target in loader:
        current = current.to(device)
        goal = goal.to(device)
        phase = phase.to(device)
        proprio = proprio.to(device)
        target = target.to(device)
        prediction = model(current, goal, phase, proprio)
        squared_error += (prediction - target).square().sum().item()
        normalized_error += (
            (prediction - target) / model.latent_std
        ).square().sum().item()
        current_error += (current - target).square().sum().item()
        goal_error += (goal - target).square().sum().item()
        elements += target.numel()
        samples += target.shape[0]
    return {
        "samples": samples,
        "latent_mse": squared_error / elements,
        "normalized_mse": normalized_error / elements,
        "current_latent_mse": current_error / elements,
        "final_goal_mse": goal_error / elements,
    }


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(base.json_value(value), output, indent=2)


def main() -> None:
    args = parse_args()
    config = base.load_config(args.config.resolve())
    settings = config["latent_subgoal_planner"]
    seed = int(config["seed"] if args.seed is None else args.seed)
    epochs = int(settings["epochs"] if args.epochs is None else args.epochs)
    device = torch.device(config["device"] if args.device is None else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    set_seed(seed)

    dataset_path = (
        config["paths"]["output_dir"]
        / "data"
        / "proprio_v2_task_continuous.npz"
    )
    lookahead_steps = int(
        settings["lookahead_steps"]
        if args.lookahead_steps is None
        else args.lookahead_steps
    )
    if lookahead_steps <= 0:
        raise ValueError("lookahead_steps must be positive.")
    samples = load_samples(dataset_path, lookahead_steps)
    train_indices, validation_indices, validation_episodes = split_by_episode(
        samples,
        float(settings["validation_fraction"]),
        seed + int(settings["split_seed_offset"]),
    )
    train_dataset = make_dataset(samples, train_indices)
    validation_dataset = make_dataset(samples, validation_indices)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=False,
    )

    current = torch.from_numpy(samples["current"][train_indices])
    goal = torch.from_numpy(samples["goal"][train_indices])
    target = torch.from_numpy(samples["target"][train_indices])
    latent_values = torch.cat([current, goal, target], dim=0)
    proprio = torch.from_numpy(samples["proprio"][train_indices])
    latent_mean = latent_values.mean(dim=0)
    latent_std = latent_values.std(dim=0).clamp_min(1e-4)
    proprio_mean = proprio.mean(dim=0)
    proprio_std = proprio.std(dim=0).clamp_min(1e-4)

    model_config = {
        "latent_dim": int(config["model"]["latent_dim"]),
        "proprio_dim": int(config["model"]["proprio_dim"]),
        "state_count": int(config["model"]["state_count"]),
        "hidden_dim": int(settings["hidden_dim"]),
        "phase_embedding_dim": int(settings["phase_embedding_dim"]),
    }
    model = LatentSubgoalPlanner(**model_config).to(device)
    model.set_normalization(
        latent_mean.to(device),
        latent_std.to(device),
        proprio_mean.to(device),
        proprio_std.to(device),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history = []
    max_patience = int(settings["early_stopping_patience"])

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
            nn.utils.clip_grad_norm_(
                model.parameters(), float(settings["max_grad_norm"])
            )
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
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(record))
        if patience >= max_patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    final_train = evaluate(
        model,
        DataLoader(
            train_dataset,
            batch_size=int(settings["batch_size"]),
            shuffle=False,
        ),
        device,
    )
    final_validation = evaluate(model, validation_loader, device)
    output_dir = config["paths"]["output_dir"]
    checkpoint_name = args.checkpoint_name
    if checkpoint_name is None:
        if lookahead_steps == int(settings["lookahead_steps"]):
            checkpoint_name = settings.get(
                "checkpoint_name", f"latent_subgoal_planner_h{lookahead_steps}.pt"
            )
        else:
            checkpoint_name = f"latent_subgoal_planner_task_only_h{lookahead_steps}.pt"
    if Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint-name must be a filename, not a path.")
    checkpoint_path = output_dir / "checkpoints" / checkpoint_name
    summary_name = (
        args.summary_name
        if args.summary_name is not None
        else f"latent_subgoal_planner_task_only_h{lookahead_steps}_summary.json"
    )
    if Path(summary_name).name != summary_name:
        raise ValueError("summary-name must be a filename, not a path.")
    summary_path = output_dir / summary_name
    for path in (checkpoint_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {path}; pass --overwrite explicitly."
            )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.model_config(),
        "model_state": best_state,
        "lookahead_steps": lookahead_steps,
        "phase": "transfer",
        "initialization": "fresh random weights; no prior checkpoint loaded",
        "seed": seed,
        "best_epoch": best_epoch,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "validation_episodes": validation_episodes,
        "dataset_path": str(dataset_path),
    }
    torch.save(payload, checkpoint_path)
    summary = {
        **payload,
        "model_state": "stored_in_checkpoint",
        "checkpoint": str(checkpoint_path),
        "samples": {
            "total": int(len(samples["current"])),
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "actual_lookahead_mean": float(
                samples["actual_lookahead"].mean()
            ),
            "actual_lookahead_25_fraction": float(
                np.mean(samples["actual_lookahead"] == lookahead_steps)
            ),
        },
        "history": history,
    }
    save_json(summary_path, summary)
    print(json.dumps(base.json_value({
        "checkpoint": checkpoint_path,
        "best_epoch": best_epoch,
        "train_metrics": final_train,
        "validation_metrics": final_validation,
        "summary": summary_path,
    }), indent=2))


if __name__ == "__main__":
    main()
