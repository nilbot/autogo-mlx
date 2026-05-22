"""Phase 5a — single-position MLX evaluator for the MCTS callback.

:class:`MLXEvaluator` is the inference surface the C++ MCTS tree calls back
into (phase 7): given a raw board and whose turn it is, return a prior over
the legal actions plus a scalar win probability for the player to move.

It is deliberately *legality-agnostic* — the caller supplies the legal action
set, because legality (captures, suicide, Ko / super-Ko) lives in the C++
``GoBoard``, not in the network. The evaluator only owns the model: one-hot
encode the board from the moving player's perspective, run a forward pass,
sigmoid the value logit, and softmax the policy logits restricted to the
legal moves.

Action indexing matches the model's flat policy head: a board move at
``(r, c)`` is index ``r * board_size + c``; the pass action is index
``board_size ** 2``. Probabilities in the returned dict are over the legal
set only and sum to 1. The value is ``P(player-to-move eventually wins)`` in
``[0, 1]`` — the same self-perspective convention the value head was trained
under (cf. :func:`autogo_mlx.loss.compute_dense_loss`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import mlx.core as mx
import numpy as np

from autogo_mlx.dataset import _one_hot_board
from autogo_mlx.model import SizeInvariantGoResNet


class MLXEvaluator:
    """Single-position network evaluator over a fixed (square) board size.

    Args:
        checkpoint_path: ``.safetensors`` weights written by
            :meth:`SizeInvariantGoResNet.save_weights` (phase 4 convention).
        board_size: Side length of the board this evaluator serves.
        channels / n_blocks / value_hidden: Architecture of the checkpointed
            net. Defaults are the production config from the upstream sweep.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        board_size: int,
        *,
        channels: int = 128,
        n_blocks: int = 10,
        value_hidden: int = 64,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)
        self.board_size = int(board_size)
        self.pass_index = self.board_size * self.board_size
        self.n_actions = self.pass_index + 1

        self.model = SizeInvariantGoResNet(
            channels=channels, n_blocks=n_blocks, value_hidden=value_hidden
        )
        self.model.load_weights(str(self.checkpoint_path))
        self.model.eval()
        mx.eval(self.model.parameters())

    def evaluate(
        self,
        board_HW: np.ndarray,
        to_play: int,
        legal_actions: Iterable[int],
    ) -> tuple[dict[int, float], float]:
        """Evaluate one position.

        Args:
            board_HW: ``(board_size, board_size)`` absolute board — ``0``
                empty, ``1`` BLACK, ``2`` WHITE (the :mod:`autogo_mlx.dataset`
                encoding).
            to_play: ``1`` (BLACK) or ``2`` (WHITE), the player to move; the
                board is one-hot encoded from this player's perspective.
            legal_actions: Flat indices of the currently legal moves — board
                cell ``r * board_size + c``, or ``board_size ** 2`` for pass.
                Typically sourced from the C++ ``GoBoard``.

        Returns:
            ``(policy, value)``. ``policy`` maps each legal action index to a
            probability (the legal set sums to 1); ``value`` is the win
            probability for ``to_play`` in ``[0, 1]``.
        """
        board_HW = np.asarray(board_HW)
        if board_HW.shape != (self.board_size, self.board_size):
            raise ValueError(
                f"board_HW shape {board_HW.shape} != "
                f"({self.board_size}, {self.board_size})"
            )
        legal = sorted({int(a) for a in legal_actions})
        if not legal:
            raise ValueError("legal_actions is empty (pass is always legal)")
        if not (0 <= legal[0] and legal[-1] < self.n_actions):
            raise ValueError(f"legal action out of range [0, {self.n_actions})")

        board_BHWC = mx.array(_one_hot_board(board_HW, to_play)[None])
        mask_BHW = mx.ones((1, self.board_size, self.board_size))
        policy_BA, value_B = self.model(board_BHWC, mask_BHW)
        mx.eval(policy_BA, value_B)

        logits_A = np.array(policy_BA[0], dtype=np.float64)
        legal_logits = logits_A[legal]
        legal_logits -= legal_logits.max()
        exp = np.exp(legal_logits)
        probs = exp / exp.sum()
        policy = {a: float(p) for a, p in zip(legal, probs)}

        value = 1.0 / (1.0 + math.exp(-float(value_B[0])))
        return policy, value
