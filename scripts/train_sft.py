#!/usr/bin/env python3
"""Ranked Human SFT Model Trainer.

Trains the SizeInvariantGoResNet model from scratch on SGF game files for a
specific target Elo bracket. Evaluates policy accuracy on validation holdouts
and backups the best checkpoints to ~/models.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.sgf_generator import SGFDataset
from autogo_mlx.loss import compute_dense_loss
from autogo_mlx.model import SizeInvariantGoResNet


def get_cosine_schedule_with_warmup(
    init_lr: float, warmup_steps: int, total_steps: int
) -> mx.array:
    """Creates a Cosine Annealing learning rate schedule with warmup.

    Args:
        init_lr: Initial learning rate.
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.

    Returns:
        A callable mapping step index to learning rate MLX array.
    """
    import math

    def lr_schedule(step: int) -> mx.array:
        if step < warmup_steps:
            return mx.array(init_lr * (step / max(1, warmup_steps)))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return mx.array(0.5 * init_lr * (1.0 + math.cos(math.pi * progress)))

    return lr_schedule


def evaluate_validation(
    model: SizeInvariantGoResNet,
    val_paths: list[Path],
    board_size: int,
    batch_size: int,
    in_channels: int,
) -> tuple[float, float, float]:
    """Runs evaluation over the validation paths.

    Args:
        model: SizeInvariantGoResNet model.
        val_paths: List of validation SGF file paths.
        board_size: Board size (9 or 19).
        batch_size: Validation batch size.
        in_channels: Number of input channels.

    Returns:
        A tuple of (average_loss, policy_loss, policy_accuracy).
    """
    model.eval()

    # Load validation generator (no shuffle, no augment)
    val_dataset = SGFDataset(val_paths, board_size=board_size, in_channels=in_channels)
    val_iter = val_dataset.iter_batches(batch_size, shuffle=False, augment=False)

    total_loss = 0.0
    total_pol_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    steps = 0

    try:
        for batch in val_iter:
            board = mx.array(batch["board_BHWC"])
            mask = mx.array(batch["mask_BHW"])
            mcts_policy = mx.array(batch["mcts_policy_BA"])
            winner = mx.array(batch["winner_B"])
            is_teacher = mx.array(batch["is_teacher_B"])

            # Compute losses
            loss, pol_loss, _, _ = compute_dense_loss(
                model, board, mask, mcts_policy, winner, is_teacher
            )

            # Compute accuracy
            policy_logits, _ = model(board, mask)
            pred_actions = mx.argmax(policy_logits, axis=-1)
            target_actions = mx.argmax(mcts_policy, axis=-1)
            correct = (pred_actions == target_actions).astype(mx.float32).sum().item()

            total_loss += loss.item()
            total_pol_loss += pol_loss.item()
            total_correct += correct
            total_samples += board.shape[0]
            steps += 1

    finally:
        val_iter.close()

    if steps == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / steps
    avg_pol_loss = total_pol_loss / steps
    accuracy = total_correct / max(total_samples, 1.0)
    
    model.train()
    return avg_loss, avg_pol_loss, accuracy


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Mugo SFT Model Trainer")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing SGF game files",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        required=True,
        help="Path to save trained weights",
    )
    parser.add_argument(
        "--bracket",
        type=int,
        required=True,
        choices=[500, 1500, 2200, 2800],
        help="Elo bracket target",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="~/models/autogo-mlx",
        help="Path to secondary backup models directory",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument(
        "--val-interval", type=int, default=200, help="Steps between validation loops"
    )
    parser.add_argument("--board-size", type=int, default=9, help="Go board size")
    parser.add_argument(
        "--in-channels",
        type=int,
        default=8,
        help="Number of input channels (3, 8, or 18)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    save_path = Path(args.save_checkpoint)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Indexing SGF dataset from {dataset_dir}...", flush=True)
    t0 = time.time()
    dataset = SGFDataset(
        dataset_dir, board_size=args.board_size, in_channels=args.in_channels
    )
    file_paths = dataset.file_paths
    print(
        f"Indexed {len(file_paths)} SGF games in {time.time() - t0:.1f}s",
        flush=True,
    )

    if len(file_paths) == 0:
        print("ERROR: No SGF files found in the dataset directory.", file=sys.stderr)
        sys.exit(1)

    # Shuffle and split into Train (90%) and Val (10%)
    random.shuffle(file_paths)
    if len(file_paths) < 2:
        train_paths = file_paths
        val_paths = file_paths
    else:
        split_idx = int(len(file_paths) * 0.9)
        if split_idx == 0:
            split_idx = 1
        elif split_idx == len(file_paths):
            split_idx = len(file_paths) - 1
        train_paths = file_paths[:split_idx]
        val_paths = file_paths[split_idx:]

    print(
        f"Data split: {len(train_paths)} train games, {len(val_paths)} validation games.",
        flush=True,
    )

    # Initialize model
    model = SizeInvariantGoResNet(
        channels=128, n_blocks=10, value_hidden=64, in_channels=args.in_channels
    )
    model.train()
    mx.eval(model.parameters())

    # Set up optimizer with Warmup + Cosine Annealing
    warmup_steps = min(200, args.steps // 5)
    lr_schedule = get_cosine_schedule_with_warmup(
        init_lr=args.lr, warmup_steps=warmup_steps, total_steps=args.steps
    )
    optimizer = opt.AdamW(learning_rate=lr_schedule, weight_decay=5e-3)

    # Define loss function
    def loss_fn(
        model: SizeInvariantGoResNet,
        board: mx.array,
        mask: mx.array,
        mcts_policy: mx.array,
        winner: mx.array,
        is_teacher: mx.array,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        loss, pol_loss, val_loss, _ = compute_dense_loss(
            model, board, mask, mcts_policy, winner, is_teacher
        )
        return loss, (pol_loss, val_loss)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Define training step
    def train_step(
        board: mx.array,
        mask: mx.array,
        mcts_policy: mx.array,
        winner: mx.array,
        is_teacher: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        (loss, (pol_loss, val_loss)), grads = loss_and_grad_fn(
            model, board, mask, mcts_policy, winner, is_teacher
        )
        optimizer.update(model, grads)

        # Policy accuracy (argmax match)
        policy_logits, _ = model(board, mask)
        pred_actions = mx.argmax(policy_logits, axis=-1)
        target_actions = mx.argmax(mcts_policy, axis=-1)
        correct = (pred_actions == target_actions).astype(mx.float32)
        accuracy = correct.mean()

        return loss, pol_loss, val_loss, accuracy

    # Start generator loop
    train_dataset = SGFDataset(train_paths, board_size=args.board_size, in_channels=args.in_channels)
    batch_iter = train_dataset.iter_batches(args.batch_size, shuffle=True, augment=True)

    print(
        f"Starting SFT training on {mx.default_device()} for {args.steps} steps...",
        flush=True,
    )
    t_train = time.time()

    losses = []
    pol_losses = []
    val_losses = []
    accuracies = []

    best_val_acc = -1.0

    for step in range(1, args.steps + 1):
        try:
            batch = next(batch_iter)
        except StopIteration:
            # Recreate generator if empty
            batch_iter.close()
            batch_iter = train_dataset.iter_batches(args.batch_size, shuffle=True, augment=True)
            batch = next(batch_iter)

        board = mx.array(batch["board_BHWC"])
        mask = mx.array(batch["mask_BHW"])
        mcts_policy = mx.array(batch["mcts_policy_BA"])
        winner = mx.array(batch["winner_B"])
        is_teacher = mx.array(batch["is_teacher_B"])

        # Execute training step
        loss, pol_loss, val_loss, accuracy = train_step(
            board, mask, mcts_policy, winner, is_teacher
        )

        mx.eval(loss, pol_loss, val_loss, accuracy, model.parameters(), optimizer.state)

        losses.append(loss.item())
        pol_losses.append(pol_loss.item())
        val_losses.append(val_loss.item())
        accuracies.append(accuracy.item())

        # Progress reporting
        if step == 1 or step % 50 == 0 or step == args.steps:
            window = min(50, step)
            roll_loss = np.mean(losses[-window:])
            roll_pol = np.mean(pol_losses[-window:])
            roll_val = np.mean(val_losses[-window:])
            roll_acc = np.mean(accuracies[-window:])
            print(
                f"Step {step:04d}/{args.steps:04d} | "
                f"Loss: {roll_loss:.4f} (Pol: {roll_pol:.4f}, Val: {roll_val:.4f}) | "
                f"Train Policy Acc: {roll_acc:.2%} | "
                f"lr: {optimizer.learning_rate.item():.6f}",
                flush=True,
            )

        # Validation loop
        if step % args.val_interval == 0 or step == args.steps:
            print("--> Running Validation...", flush=True)
            val_loss, val_pol, val_acc = evaluate_validation(
                model=model,
                val_paths=val_paths,
                board_size=args.board_size,
                batch_size=args.batch_size,
                in_channels=args.in_channels,
            )
            print(
                f"    [Val] Loss: {val_loss:.4f} (Pol: {val_pol:.4f}) | "
                f"Validation Policy Acc: {val_acc:.2%}",
                flush=True,
            )

            # Check if this is the best checkpoint based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_weights(str(save_path))
                print(
                    f"    🟢 NEW BEST validation accuracy! Saved checkpoint to {save_path}",
                    flush=True,
                )

                # Replicate to backup folder dynamically
                if args.backup_dir:
                    try:
                        backup_base = Path(args.backup_dir).expanduser()
                        backup_dir = (
                            backup_base
                            / "sft"
                            / f"board_{args.board_size}x{args.board_size}"
                            / f"{args.bracket}_elo"
                        )
                        backup_dir.mkdir(parents=True, exist_ok=True)
                        backup_path = backup_dir / f"sft_{args.bracket}.safetensors"
                        
                        # Save weights to backup path
                        model.save_weights(str(backup_path))
                        print(f"    Replicated backup checkpoint to {backup_path}", flush=True)
                    except Exception as e:
                        print(
                            f"    Warning: Failed to replicate backup to {args.backup_dir}: {e}",
                            flush=True,
                        )

    batch_iter.close()
    duration = time.time() - t_train
    print(
        f"\nSFT Training completed in {duration:.1f}s ({duration/args.steps:.3f}s/step).",
        flush=True,
    )
    print(f"Best Validation Policy Accuracy: {best_val_acc:.2%}", flush=True)


if __name__ == "__main__":
    main()
