"""Phase 10 — Random agent for bootstrapping and evaluations."""

from __future__ import annotations

import random
from typing import Tuple, Union, Iterable

import numpy as np

from autogo_mlx.cpp_bridge import GoBoard


class RandomAgent:
    """Agent that plays random legal moves on the GoBoard."""

    def __init__(self, board_size: int = 9) -> None:
        self.board_size = board_size
        self.pass_index = board_size * board_size
        self.last_mcts_policy: np.ndarray | None = None

    def start_game(self, board_size: int) -> None:
        self.board_size = board_size
        self.pass_index = board_size * board_size

    def end_game(self) -> None:
        pass

    def select_move(
        self,
        board: GoBoard,
        seed: int | None = None,
        legal_actions: Iterable[int] | None = None,
        as_flat: bool = False,
    ) -> Union[Tuple[int, int], int]:
        """Select a random legal move.

        Args:
            board: Current board state.
            seed: Optional seed for reproducibility.
            legal_actions: Optional legal action filter.
            as_flat: Whether to return the flat index instead of (row, col) coordinates.

        Returns:
            Coordinates (row, col) or PASS=(-1, -1), or flat index when as_flat=True.
        """
        if seed is not None:
            random.seed(seed)

        legal_flat = board.get_legal_moves_flat()
        legal_actions_list = legal_flat + [self.pass_index]

        # Choose a random legal action flat index
        flat_idx = random.choice(legal_actions_list)

        # Fallback dense policy (one-hot distribution on the played action)
        n_actions = self.board_size * self.board_size + 1
        policy = np.zeros(n_actions, dtype=np.float32)
        policy[flat_idx] = 1.0
        self.last_mcts_policy = policy

        if as_flat or legal_actions is not None:
            return flat_idx

        if flat_idx == self.pass_index:
            return (-1, -1)
        return board.row_col(flat_idx)

    def close(self) -> None:
        pass
