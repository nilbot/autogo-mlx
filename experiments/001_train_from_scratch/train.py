#!/usr/bin/env python3
"""Phase 10 — Compiled MLX training script.

Trains the SizeInvariantGoResNet model for 2,000 steps on collected self-play games.
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
    parser = argparse.ArgumentParser(description="Mugo Phase 10 Model Trainer")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing .npz game files",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Checkpoint to resume/load weights from (empty to train from scratch)",
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
        "--steps", type=int, default=2000, help="Number of training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--in-channels",
        type=int,
        default=8,
        help="Number of input channels (3 for absolute, 8 for liberties+Ko, 18 for history)",
    )
    parser.add_argument(
        "--value-lambda",
        type=float,
        default=0.5,
        help="Weight for final outcome in blended value targets (1.0 = pure outcome, 0.0 = pure search value)",
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    save_path = Path(args.save_checkpoint)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {dataset_dir}...", flush=True)
    t0 = time.time()
    dataset = GoDataset(
        dataset_dir, board_size=9, in_memory=True, in_channels=args.in_channels
    )
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

    # Initialize model
    model = SizeInvariantGoResNet(
        channels=128, n_blocks=10, value_hidden=64, in_channels=args.in_channels
    )

    # Load weights if resume-from is provided and non-empty (strict=False)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"Resuming weights from checkpoint: {resume_path} (strict=False)", flush=True)
            model.load_weights(str(resume_path), strict=False)
        else:
            print(
                f"ERROR: Resume checkpoint specified but not found: {resume_path}",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(
            "No resume checkpoint specified. Training from scratch (random initialization)...",
            flush=True,
        )

    model.train()
    mx.eval(model.parameters())

    # Set up optimizer with Cosine Annealing + Warmup schedule
    warmup_steps = 200
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
        final_score: mx.array | None = None,
        ownership_target: mx.array | None = None,
        has_ownership: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array]]:
        loss, pol_loss, val_loss, own_loss = compute_dense_loss(
            model,
            board,
            mask,
            mcts_policy,
            winner,
            is_teacher,
            score_target_B=final_score,
            ownership_target_BHW=ownership_target,
            has_ownership_target_B=has_ownership,
        )
        return loss, (pol_loss, val_loss, own_loss)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    # Training step
    def train_step(
        board: mx.array,
        mask: mx.array,
        mcts_policy: mx.array,
        winner: mx.array,
        is_teacher: mx.array,
        final_score: mx.array | None,
        ownership_target: mx.array | None,
        has_ownership: mx.array | None,
        step: int,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        (loss, (pol_loss, val_loss, own_loss)), grads = loss_and_grad_fn(
            model,
            board,
            mask,
            mcts_policy,
            winner,
            is_teacher,
            final_score,
            ownership_target,
            has_ownership,
        )

        # Head Warm-Up Phase: freeze trunk and policy during first 1000 steps
        if step <= 1000:
            def freeze_gradients_recursive(grads_dict, prefix=""):
                if isinstance(grads_dict, dict):
                    for k, v in list(grads_dict.items()):
                        full_key = f"{prefix}.{k}" if prefix else k
                        if isinstance(v, (dict, list)):
                            freeze_gradients_recursive(v, full_key)
                        else:
                            if not any(h in full_key for h in ["value", "score", "ownership"]):
                                grads_dict[k] = mx.zeros_like(v)
                elif isinstance(grads_dict, list):
                    for i, v in enumerate(grads_dict):
                        full_key = f"{prefix}[{i}]"
                        if isinstance(v, (dict, list)):
                            freeze_gradients_recursive(v, full_key)
                        else:
                            if not any(h in full_key for h in ["value", "score", "ownership"]):
                                grads_dict[i] = mx.zeros_like(v)

            freeze_gradients_recursive(grads)

        optimizer.update(model, grads)

        # Calculate policy accuracy (teacher samples only)
        policy_logits, _ = model(board, mask)
        pred_actions = mx.argmax(policy_logits, axis=-1)
        target_actions = mx.argmax(mcts_policy, axis=-1)
        correct = (pred_actions == target_actions).astype(mx.float32)
        accuracy = (correct * is_teacher).sum() / mx.maximum(is_teacher.sum(), 1.0)

        # Calculate score MAE if score prediction head is active
        if final_score is not None:
            _, _, score_logits = model(board, mask, return_score=True)
            score_mae = mx.mean(mx.abs(score_logits - final_score))
        else:
            score_mae = mx.array(0.0)

        return loss, pol_loss, val_loss, own_loss, accuracy, score_mae

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
    ownership_losses = []
    accuracies = []
    score_maes = []

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
        final_score = (
            mx.array(batch["final_score_B"]) if "final_score_B" in batch else None
        )
        ownership_target = (
            mx.array(batch["ownership_target_BHW"]) if "ownership_target_BHW" in batch else None
        )
        has_ownership = (
            mx.array(batch["has_ownership_target_B"]) if "has_ownership_target_B" in batch else None
        )
        
        # Blend winner and search Q-values using value_lambda
        # winner is self-perspective outcome {0, 1}. root_q_value_B is self-perspective search Q-value [0, 1].
        if "root_q_value_B" in batch and args.value_lambda < 1.0:
            root_q = mx.array(batch["root_q_value_B"])
            value_target = (1.0 - args.value_lambda) * root_q + args.value_lambda * winner
        else:
            value_target = winner

        # Execute step
        loss, pol_loss, val_loss, own_loss, accuracy, score_mae = train_step(
            board,
            mask,
            mcts_policy,
            value_target,
            is_teacher,
            final_score,
            ownership_target,
            has_ownership,
            step,
        )

        # Force evaluation of the returned scalars and state updates
        mx.eval(
            loss,
            pol_loss,
            val_loss,
            own_loss,
            accuracy,
            score_mae,
            model.parameters(),
            optimizer.state,
        )

        losses.append(loss.item())
        policy_losses.append(pol_loss.item())
        value_losses.append(val_loss.item())
        ownership_losses.append(own_loss.item())
        accuracies.append(accuracy.item())
        score_maes.append(score_mae.item())

        if step == 1 or step % 250 == 0 or step == args.steps:
            # Calculate rolling average over the last 100 steps to smooth out batch-level noise
            window = min(100, step)
            roll_loss = np.mean(losses[-window:])
            roll_pol = np.mean(policy_losses[-window:])
            roll_val = np.mean(value_losses[-window:])
            roll_own = np.mean(ownership_losses[-window:])
            roll_acc = np.mean(accuracies[-window:])
            roll_mae = np.mean(score_maes[-window:])

            score_log = f", Score MAE: {roll_mae:.2f}" if args.in_channels == 18 else ""
            print(
                f"Step {step:04d}/{args.steps:04d} | "
                f"Loss: {roll_loss:.4f} (Pol: {roll_pol:.4f}, Val: {roll_val:.4f}, Own: {roll_own:.4f}) | "
                f"Train Policy Acc: {roll_acc:.2%}{score_log} | "
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
    avg_loss = np.mean(losses[-100:])
    avg_pol = np.mean(policy_losses[-100:])
    avg_val = np.mean(value_losses[-100:])
    avg_own = np.mean(ownership_losses[-100:])
    avg_acc = np.mean(accuracies[-100:])
    avg_mae = np.mean(score_maes[-100:])
    print("\nFinal average metrics over last 100 steps:", flush=True)
    print(
        f"  Loss: {avg_loss:.4f} (Pol: {avg_pol:.4f}, Val: {avg_val:.4f}, Own: {avg_own:.4f})", flush=True
    )
    print(f"  Policy Acc: {avg_acc:.2%}", flush=True)
    if args.in_channels == 18:
        print(f"  Score MAE: {avg_mae:.2f}", flush=True)


if __name__ == "__main__":
    main()
