"""Phase 7c — End-to-end self-play smoke test.

Loads a randomly initialized MLX model checkpoint, plays one 9x9 game
between two MLXNNMCTSAgents with a small simulation limit (n_simulations=16),
asserts successful termination, saves the game to NPZ, and verifies that the
NPZ round-trips perfectly through GoDataset.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mugo.agents.nn_mcts import MLXNNMCTSAgent
from mugo.batched_inference import BatchedMLXEvaluator
from mugo.dataset import GoDataset
from mugo.gameplay import play_game, save_game_data
from mugo.inference import MLXEvaluator
from mugo.model import SizeInvariantGoResNet


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Fixture to create a random SizeInvariantGoResNet model and save its weights."""
    mx.random.seed(42)
    model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)
    checkpoint_path = tmp_path / "dummy_model.safetensors"
    model.save_weights(str(checkpoint_path))
    return checkpoint_path


def test_selfplay_smoke_single_evaluator(dummy_checkpoint: Path, tmp_path: Path) -> None:
    board_size = 9
    evaluator = MLXEvaluator(dummy_checkpoint, board_size)

    # 1. Create two agents using the single evaluator
    black_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=16,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        temperature=1.0,
    )
    white_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=16,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        temperature=1.0,
    )

    try:
        # 2. Run a short self-play game
        # Max moves is limited to 15 to keep the test quick, but long enough to verify search logic
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            max_moves=15,
            seed=42,
        )

        assert record.board_size == board_size
        assert record.num_moves > 0
        assert len(record.moves) == record.num_moves
        assert len(record.boards) == record.num_moves
        assert len(record.mcts_policies) == record.num_moves
        assert record.winner in (0, 1, 2)
        assert record.termination in ("double_pass", "max_moves")

        # 3. Save the GameRecord to NPZ
        npz_path = tmp_path / "selfplay_game_single.npz"
        save_game_data(record, npz_path)
        assert npz_path.exists()

        # 4. Roundtrip through GoDataset
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        # Move the npz to the dataset folder
        npz_dest = dataset_dir / "game_0001.npz"
        npz_path.rename(npz_dest)

        ds = GoDataset(dataset_dir, board_size=board_size)
        assert len(ds) == record.num_moves

        # Inspect the first sample
        sample = ds[0]
        assert "board" in sample
        board = sample["board"]
        assert isinstance(board, np.ndarray)
        assert board.shape == (board_size, board_size)

        assert "mask" in sample
        mask = sample["mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (board_size, board_size)

        assert "mcts_policy" in sample
        mcts_policy = sample["mcts_policy"]
        assert isinstance(mcts_policy, np.ndarray)
        assert mcts_policy.shape == (board_size * board_size + 1,)

        assert "winner" in sample
        assert "is_teacher" in sample

        # Check shapes and sum of policy
        np.testing.assert_allclose(mcts_policy.sum(), 1.0, atol=1e-5)

    finally:
        black_agent.close()
        white_agent.close()


def test_selfplay_smoke_batched_evaluator(dummy_checkpoint: Path, tmp_path: Path) -> None:
    board_size = 9
    evaluator = BatchedMLXEvaluator(dummy_checkpoint, board_size, batch_size=8, timeout_ms=2.0)

    # 1. Create two agents using the batched evaluator
    black_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=16,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        temperature=1.0,
        leaf_batch_size=4,
    )
    white_agent = MLXNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=16,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        temperature=1.0,
        leaf_batch_size=4,
    )

    try:
        # 2. Run a short self-play game
        # Max moves is limited to 15 to keep the test quick, but long enough to verify search logic
        record = play_game(
            black_agent=black_agent,
            white_agent=white_agent,
            board_size=board_size,
            max_moves=15,
            seed=101,
        )

        assert record.board_size == board_size
        assert record.num_moves > 0
        assert len(record.moves) == record.num_moves
        assert len(record.boards) == record.num_moves
        assert len(record.mcts_policies) == record.num_moves
        assert record.winner in (0, 1, 2)
        assert record.termination in ("double_pass", "max_moves")

        # 3. Save the GameRecord to NPZ
        npz_path = tmp_path / "selfplay_game_batched.npz"
        save_game_data(record, npz_path)
        assert npz_path.exists()

        # 4. Roundtrip through GoDataset
        dataset_dir = tmp_path / "dataset_batched"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        # Move the npz to the dataset folder
        npz_dest = dataset_dir / "game_0001.npz"
        npz_path.rename(npz_dest)

        ds = GoDataset(dataset_dir, board_size=board_size)
        assert len(ds) == record.num_moves

        # Inspect the first sample
        sample = ds[0]
        assert "board" in sample
        board = sample["board"]
        assert isinstance(board, np.ndarray)
        assert board.shape == (board_size, board_size)

        assert "mask" in sample
        mask = sample["mask"]
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (board_size, board_size)

        assert "mcts_policy" in sample
        mcts_policy = sample["mcts_policy"]
        assert isinstance(mcts_policy, np.ndarray)
        assert mcts_policy.shape == (board_size * board_size + 1,)

        assert "winner" in sample
        assert "is_teacher" in sample

        # Check shapes and sum of policy
        np.testing.assert_allclose(mcts_policy.sum(), 1.0, atol=1e-5)

    finally:
        black_agent.close()
        white_agent.close()
        evaluator.close()
