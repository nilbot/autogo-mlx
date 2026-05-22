"""Phase 7a — Neural Network Monte Carlo Tree Search Agent in MLX.

Combines MLX network evaluations with native C++ MCTS search tree traversal.
Supports both single-position and dynamically-batched leaf-parallel search.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np

from autogo_mlx.cpp_bridge import (
    PASS_ACTION,
    GoBoard,
    MCTSConfig,
    MCTSTree,
)

if TYPE_CHECKING:
    from autogo_mlx.batched_inference import BatchedMLXEvaluator
    from autogo_mlx.inference import MLXEvaluator


class MLXNNMCTSAgent:
    """Agent that plays Go using MCTS driven by an MLX policy/value network.

    It constructs a C++ ``MCTSTree`` internally and runs simulations on it,
    calling back into Python to evaluate leaf nodes using either a single or
    batched MLX evaluator.
    """

    def __init__(
        self,
        evaluator: Union[MLXEvaluator, BatchedMLXEvaluator],
        n_simulations: int = 16,
        *,
        c_puct: float = 1.0,
        dirichlet_alpha: float = 0.0,
        temperature: float = 1.0,
        leaf_batch_size: int | None = None,
    ) -> None:
        """Initialize the MLX NN MCTS Agent.

        Args:
            evaluator: The MLX policy/value network evaluator.
            n_simulations: Number of MCTS simulations per move (default: 16).
            c_puct: Exploration constant for MCTS selection (default: 1.0).
            dirichlet_alpha: Dirichlet noise alpha at the root (0.0 to disable).
            temperature: Temperature for visit probabilities / move selection.
            leaf_batch_size: Leaf batch size for parallel simulations.
                Defaults to 8 if evaluator is batched, 0 otherwise.
        """
        self.evaluator = evaluator
        self.n_simulations = int(n_simulations)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.temperature = float(temperature)

        self.board_size = self.evaluator.board_size
        self.pass_index = self.board_size * self.board_size

        # Auto-detect batched execution
        if leaf_batch_size is None:
            # Import dynamically to avoid circular import if needed
            from autogo_mlx.batched_inference import BatchedMLXEvaluator

            if isinstance(self.evaluator, BatchedMLXEvaluator):
                self.leaf_batch_size = 8
            else:
                self.leaf_batch_size = 0
        else:
            self.leaf_batch_size = int(leaf_batch_size)

        self.last_mcts_policy: np.ndarray | None = None

    def close(self) -> None:
        """No-op. Native C++ MCTS batching eliminates the Python thread pool."""
        pass

    def __enter__(self) -> MLXNNMCTSAgent:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def select_move(
        self,
        board: GoBoard,
        seed: int | None = None,
        legal_actions: Iterable[int] | None = None,
        as_flat: bool = False,
    ) -> Union[Tuple[int, int], int]:
        """Select a move using MCTS.

        Args:
            board: Current board state (C++ GoBoard).
            seed: Optional seed for Dirichlet noise reproducibility.
            legal_actions: Optional legal action filter (for raw action selection).
            as_flat: When True, return flat index (0 to board_size^2) instead of
                (row, col) coordinates.

        Returns:
            Coordinates (row, col) or PASS=(-1, -1), or flat index when as_flat=True.
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Setup MCTS Search Config
        config = MCTSConfig()
        config.c_puct = self.c_puct
        config.dirichlet_alpha = self.dirichlet_alpha
        config.dirichlet_weight = 0.25 if self.dirichlet_alpha > 0.0 else 0.0
        config.temperature = self.temperature
        config.lambda_ = 0.0  # pure value network, no rollout

        tree = MCTSTree(board, config)

        # 2. Define the Python Callback for MCTS Leaves
        def single_evaluator_cb(state: GoBoard) -> Tuple[Dict[int, float], float]:
            """Process single leaf evaluation."""
            board_HW = state.to_numpy()
            to_play = state.to_play()
            legal_flat = state.get_legal_moves_flat()
            legal_actions_nn = legal_flat + [self.pass_index]

            policy_nn, value_nn = self.evaluator.evaluate(
                board_HW, to_play, legal_actions_nn
            )

            # Map NN's pass index back to C++ PASS_ACTION (-1)
            policy_cpp = {
                (a if a < self.pass_index else PASS_ACTION): p
                for a, p in policy_nn.items()
            }
            return policy_cpp, value_nn

        def batched_evaluator_cb(
            states: List[GoBoard],
        ) -> List[Tuple[Dict[int, float], float]]:
            """Process a batch of leaf evaluations directly in a single forward pass."""
            eval_inputs = []
            for state in states:
                board_HW = state.to_numpy()
                to_play = state.to_play()
                legal_flat = state.get_legal_moves_flat()
                legal_actions_nn = legal_flat + [self.pass_index]
                eval_inputs.append((board_HW, to_play, legal_actions_nn))

            results_nn = self.evaluator.evaluate_batch(eval_inputs)

            results: List[Tuple[Dict[int, float], float]] = []
            for policy_nn, value_nn in results_nn:
                policy_cpp = {
                    (a if a < self.pass_index else PASS_ACTION): p
                    for a, p in policy_nn.items()
                }
                results.append((policy_cpp, value_nn))
            return results

        # 3. Run search simulations
        if self.leaf_batch_size > 0:
            tree.run_simulations_batched(
                self.n_simulations, self.leaf_batch_size, batched_evaluator_cb
            )
        else:
            tree.run_simulations(self.n_simulations, single_evaluator_cb)

        # 4. Extract action probabilities and selected move
        probs_cpp = tree.get_action_probabilities(self.temperature)

        # Build dense policy distribution over actions for GameRecord
        n_actions = self.board_size * self.board_size + 1
        dense_policy = np.zeros(n_actions, dtype=np.float32)
        for act_idx, prob in probs_cpp.items():
            if act_idx == PASS_ACTION:
                dense_policy[-1] = prob
            else:
                dense_policy[act_idx] = prob

        # Ensure it sums to exactly 1.0
        total_p = dense_policy.sum()
        if total_p > 0:
            dense_policy /= total_p
        self.last_mcts_policy = dense_policy

        # Choose the final action index
        action_idx = tree.select_action(self.temperature)

        # 5. Return coordinate or flat representation
        if as_flat or legal_actions is not None:
            # If the user requested flat action index
            if action_idx == PASS_ACTION:
                return self.pass_index
            return action_idx
        else:
            # Return coordinate representation (row, col)
            if action_idx == PASS_ACTION:
                return (-1, -1)
            return board.row_col(action_idx)
