#!/usr/bin/env python3
"""Phase 8a — Compiled MLX training script.

Trains the SizeInvariantGoResNet model for ~300 steps on collected self-play games.
Uses compiled step execution and AdamW optimization.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

# Ensure we import from autogo_mlx correctly
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.dataset import GoDataset
from autogo_mlx.loss import compute_dense_loss
from autogo_mlx.model import SizeInvariantGoResNet


def get_cosine_schedule_with_warmup(
    init_lr: float, warmup_steps: int, total_steps: int
):
    import math

    def lr_schedule(step: int):
        if step < warmup_steps:
            return mx.array(init_lr * (step / max(1, warmup_steps)))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return mx.array(0.5 * init_lr * (1.0 + math.cos(math.pi * progress)))

    return lr_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Mugo Phase 8a Model Trainer")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing .npz game files",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        required=True,
        help="Checkpoint to resume/load weights from",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=str,
        required=True,
        help="Path to save trained weights",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--steps", type=int, default=300, help="Number of training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    resume_path = Path(args.resume_from)
    save_path = Path(args.save_checkpoint)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume_path.exists():
        print(f"ERROR: Resume checkpoint not found: {resume_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {dataset_dir}...", flush=True)
    t0 = time.time()
    dataset = GoDataset(dataset_dir, board_size=9, in_memory=True)
    print(
        f"Loaded {len(dataset)} positions from dataset in {time.time() - t0:.1f}s",
        flush=True,
    )

    if len(dataset) < args.batch_size:
        print(
            f"ERROR: Dataset has only {len(dataset)} positions, which is less than batch size {args.batch_size}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialize model and load weights
    model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)
    model.load_weights(str(resume_path))
    model.train()
    mx.eval(model.parameters())

    # Set up optimizer with Cosine Annealing + Warmup schedule
    warmup_steps = min(200, args.steps // 3)
    lr_schedule = get_cosine_schedule_with_warmup(
        init_lr=args.lr, warmup_steps=warmup_steps, total_steps=args.steps
    )
    optimizer = opt.AdamW(learning_rate=lr_schedule, weight_decay=5e-3)

    # Loss and gradient function
    def loss_fn(
        model: SizeInvariantGoResNet,
        board: mx.array,
        mask: mx.array,
        mcts_policy: mx.array,
        winner: mx.array,
        is_teacher: mx.array,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        loss, pol_loss, val_loss = compute_dense_loss(
            model, board, mask, mcts_policy, winner, is_teacher
        )
        return loss, (pol_loss, val_loss)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training step
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

        # Calculate policy accuracy (teacher samples only)
        policy_logits, _ = model(board, mask)
        pred_actions = mx.argmax(policy_logits, axis=-1)
        target_actions = mx.argmax(mcts_policy, axis=-1)
        correct = (pred_actions == target_actions).astype(mx.float32)
        accuracy = (correct * is_teacher).sum() / mx.maximum(is_teacher.sum(), 1.0)

        return loss, pol_loss, val_loss, accuracy

    print(
        f"Starting training on {mx.default_device()} for {args.steps} steps...",
        flush=True,
    )
    t_train = time.time()

    # We will draw batches continuously from an infinite generator loop
    batch_iter = dataset.iter_batches(args.batch_size, shuffle=True, augment=True)

    losses = []
    policy_losses = []
    value_losses = []
    accuracies = []

    for step in range(1, args.steps + 1):
        try:
            batch = next(batch_iter)
        except StopIteration:
            # Recreate generator if we run out of positions
            batch_iter = dataset.iter_batches(
                args.batch_size, shuffle=True, augment=True
            )
            batch = next(batch_iter)

        # Convert to MLX arrays
        board = mx.array(batch["board_BHWC"])
        mask = mx.array(batch["mask_BHW"])
        mcts_policy = mx.array(batch["mcts_policy_BA"])
        winner = mx.array(batch["winner_B"])
        is_teacher = mx.array(batch["is_teacher_B"])

        # Execute step
        loss, pol_loss, val_loss, accuracy = train_step(
            board, mask, mcts_policy, winner, is_teacher
        )

        # Force evaluation of the returned scalars and state updates
        mx.eval(loss, pol_loss, val_loss, accuracy, model.parameters(), optimizer.state)

        losses.append(loss.item())
        policy_losses.append(pol_loss.item())
        value_losses.append(val_loss.item())
        accuracies.append(accuracy.item())

        if step == 1 or step % 50 == 0:
            print(
                f"Step {step:03d}/{args.steps:03d} | "
                f"Loss: {loss.item():.4f} (Pol: {pol_loss.item():.4f}, Val: {val_loss.item():.4f}) | "
                f"Train Policy Acc: {accuracy.item():.2%} | "
                f"lr: {optimizer.learning_rate.item():.6f}",
                flush=True,
            )

    train_duration = time.time() - t_train
    print(
        f"Training finished in {train_duration:.1f}s ({train_duration / args.steps:.3f}s/step).",
        flush=True,
    )

    # Save model weights
    model.save_weights(str(save_path))
    print(f"Saved checkpoint to {save_path}", flush=True)

    # Print summary metrics
    avg_loss = np.mean(losses[-50:])
    avg_pol = np.mean(policy_losses[-50:])
    avg_val = np.mean(value_losses[-50:])
    avg_acc = np.mean(accuracies[-50:])
    print("\nFinal average metrics over last 50 steps:", flush=True)
    print(
        f"  Loss: {avg_loss:.4f} (Pol: {avg_pol:.4f}, Val: {avg_val:.4f})", flush=True
    )
    print(f"  Policy Acc: {avg_acc:.2%}", flush=True)


if __name__ == "__main__":
    main()
