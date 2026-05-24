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

from autogo_mlx.dataset import _one_hot_board, _compute_liberties_numpy
from autogo_mlx.model import SizeInvariantGoResNet


def _find_ko_point_evaluator(board_HW: np.ndarray, to_play: int, legal_set: set[int]) -> np.ndarray:
    """Finds Ko point purely from current board state and legal actions."""
    h, w = board_HW.shape
    ko_plane = np.zeros((h, w), dtype=np.float32)
    opp_color = 2 if to_play == 1 else 1
    for r in range(h):
        for c in range(w):
            if board_HW[r, c] == 0:
                idx = r * w + c
                if idx not in legal_set:
                    surrounded = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if board_HW[nr, nc] != opp_color:
                                surrounded = False
                                break
                    if surrounded:
                        ko_plane[r, c] = 1.0
    return ko_plane



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
        in_channels: int = 8,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(self.checkpoint_path)
        self.board_size = int(board_size)
        self.pass_index = self.board_size * self.board_size
        self.n_actions = self.pass_index + 1
        self.in_channels = int(in_channels)

        self.model = SizeInvariantGoResNet(
            channels=channels, n_blocks=n_blocks, value_hidden=value_hidden, in_channels=in_channels
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

        if self.in_channels == 8:
            lib_1, lib_2, lib_3, lib_4 = _compute_liberties_numpy(board_HW)
            ko = _find_ko_point_evaluator(board_HW, to_play, set(legal))
            
            one_hot = _one_hot_board(board_HW, to_play)
            board_8ch = np.zeros((self.board_size, self.board_size, 8), dtype=np.float32)
            board_8ch[..., :3] = one_hot
            board_8ch[..., 3] = lib_1
            board_8ch[..., 4] = lib_2
            board_8ch[..., 5] = lib_3
            board_8ch[..., 6] = lib_4
            board_8ch[..., 7] = ko
            
            board_BHWC = mx.array(board_8ch[None])
        else:
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
