"""Phase 4 — synthetic-data overfit run.

The "the recipe wires together" milestone. We generate ~1000 random Go-board
positions with deterministic policy and value targets, then train the
:class:`SizeInvariantGoResNet` for 500 steps and assert it overfits the
training set to ≥ 80% policy accuracy. A checkpoint is written via
``model.save_weights`` and reloaded into a fresh model to verify the
safetensors round-trip survives layout/dtype conversion.

The synthetic targets are a deterministic function of the input so we know
the task is in-distribution by construction:

* **policy target**: argmax over flat cells of ``self_channel * row_idx -
  opponent_channel * col_idx`` plus a tiny tiebreak so the argmax is unique
  on all-empty boards. One-hot at that cell; the pass slot is never used
  (the task is "find the special cell"). Plenty of variety across 1024
  random boards.
* **value target**: ``1`` if the position has more BLACK stones than
  WHITE, else ``0``. The current player is always BLACK in this script, so
  the value head learns "did I have more stones at this snapshot?".

Both labels are functions of low-order moments of the input, so a 10-block
ResNet should easily memorise them inside 500 steps of batch-64 AdamW.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tyro

from mugo.loss import compute_dense_loss
from mugo.model import SizeInvariantGoResNet


def make_synthetic(
    n: int, board_size: int, seed: int
) -> dict[str, np.ndarray]:
    """Build ``n`` synthetic positions with deterministic labels (numpy host)."""
    rng = np.random.default_rng(seed)
    bs = board_size
    raw = rng.integers(0, 3, size=(n, bs, bs), dtype=np.int8)

    board_NHWC = np.zeros((n, bs, bs, 3), dtype=np.float32)
    board_NHWC[..., 0] = raw == 0
    board_NHWC[..., 1] = raw == 1  # self = BLACK
    board_NHWC[..., 2] = raw == 2  # opponent = WHITE

    row_idx = (np.arange(bs, dtype=np.float32) + 1)[:, None]
    col_idx = (np.arange(bs, dtype=np.float32) + 1)[None, :]
    score = board_NHWC[..., 1] * row_idx - board_NHWC[..., 2] * col_idx
    score = score + 1e-3 * np.arange(bs * bs, dtype=np.float32).reshape(bs, bs)
    target_cell = score.reshape(n, bs * bs).argmax(axis=-1).astype(np.int32)

    policy_NA = np.zeros((n, bs * bs + 1), dtype=np.float32)
    policy_NA[np.arange(n), target_cell] = 1.0

    black = (raw == 1).sum(axis=(1, 2))
    white = (raw == 2).sum(axis=(1, 2))
    winner_N = (black > white).astype(np.float32)

    return {
        "board_NHWC": board_NHWC,
        "mask_NHW": np.ones((n, bs, bs), dtype=np.float32),
        "policy_NA": policy_NA,
        "winner_N": winner_N,
        "is_teacher_N": np.ones((n,), dtype=np.float32),
        "target_cell_N": target_cell,
    }


def _accuracy(
    model: SizeInvariantGoResNet,
    board: mx.array,
    mask: mx.array,
    target: np.ndarray,
    chunk: int = 256,
) -> float:
    n = board.shape[0]
    correct = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        logits, _ = model(board[s:e], mask[s:e])
        mx.eval(logits)
        correct += int((np.array(logits.argmax(axis=-1)) == target[s:e]).sum())
    return correct / n


@dataclass
class Config:
    n_samples: int = 1024
    board_size: int = 9
    batch_size: int = 64
    n_steps: int = 500
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    accuracy_threshold: float = 0.80
    checkpoint: Path = Path("checkpoints/synthetic_overfit.safetensors")
    log_every: int = 50


def main(cfg: Config = Config()) -> None:
    assert mx.metal.is_available(), "Metal GPU required for meaningful training"
    mx.set_default_device(mx.gpu)  # type: ignore[arg-type]
    mx.random.seed(cfg.seed)

    data = make_synthetic(cfg.n_samples, cfg.board_size, seed=cfg.seed)
    board = mx.array(data["board_NHWC"])
    mask = mx.array(data["mask_NHW"])
    policy = mx.array(data["policy_NA"])
    winner = mx.array(data["winner_N"])
    is_teacher = mx.array(data["is_teacher_N"])
    target_np = data["target_cell_N"]

    model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)

    def loss_fn(
        m: SizeInvariantGoResNet,
        b: mx.array, mk: mx.array, p: mx.array, w: mx.array, t: mx.array,
    ) -> mx.array:
        return compute_dense_loss(m, b, mk, p, w, t)[0]

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.AdamW(learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay)

    rng = np.random.default_rng(cfg.seed + 1)
    t0 = time.perf_counter()
    for step in range(1, cfg.n_steps + 1):
        idx = mx.array(rng.integers(0, cfg.n_samples, size=cfg.batch_size))
        loss, grads = loss_and_grad(
            model, board[idx], mask[idx], policy[idx], winner[idx], is_teacher[idx]
        )
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        if step == 1 or step % cfg.log_every == 0:
            print(
                f"step {step:>4d}  loss={float(loss):.4f}  "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )

    acc = _accuracy(model, board, mask, target_np)
    print(f"final policy accuracy = {acc:.4f}  (n={cfg.n_samples})")

    cfg.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(cfg.checkpoint))
    print(f"saved checkpoint → {cfg.checkpoint}")

    reloaded = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)
    reloaded.load_weights(str(cfg.checkpoint))
    mx.eval(reloaded.parameters())
    probe_b, probe_m = board[:32], mask[:32]
    p1, v1 = model(probe_b, probe_m)
    p2, v2 = reloaded(probe_b, probe_m)
    mx.eval(p1, v1, p2, v2)
    policy_err = float(mx.max(mx.abs(p1 - p2)).item())
    value_err = float(mx.max(mx.abs(v1 - v2)).item())
    print(f"reload max|Δpolicy|={policy_err:.2e}  max|Δvalue|={value_err:.2e}")
    assert policy_err < 1e-5 and value_err < 1e-5, "checkpoint roundtrip mismatch"

    if acc < cfg.accuracy_threshold:
        raise SystemExit(
            f"final policy accuracy {acc:.4f} < threshold {cfg.accuracy_threshold:.2f}"
        )


if __name__ == "__main__":
    main(tyro.cli(Config))
