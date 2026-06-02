import pytest
import numpy as np
import mlx.core as mx
from pathlib import Path

from autogo_mlx.cpp_bridge import GoBoard
from autogo_mlx.gameplay import play_vectorized_games, save_game_data, GameRecord
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.inference import MLXEvaluator
from autogo_mlx.dataset import GoDataset
from autogo_mlx.model import SizeInvariantGoResNet

@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a dummy SizeInvariantGoResNet weights file for 18 channels."""
    mx.random.seed(42)
    model = SizeInvariantGoResNet(channels=32, n_blocks=2, value_hidden=16, in_channels=18)
    checkpoint_path = tmp_path / "dummy_18ch.safetensors"
    model.save_weights(str(checkpoint_path))
    return checkpoint_path


def test_evaluator_18ch_history_plane_alignment(dummy_checkpoint: Path):
    """Verify that MLXEvaluator and BatchedMLXEvaluator produce matching, correct 18ch features."""
    board_size = 9
    evaluator_single = MLXEvaluator(dummy_checkpoint, board_size, channels=32, n_blocks=2, value_hidden=16, in_channels=18)
    evaluator_batch = BatchedMLXEvaluator(dummy_checkpoint, board_size, batch_size=2, timeout_ms=2.0, channels=32, n_blocks=2, value_hidden=16, in_channels=18)

    # Construct some synthetic board history
    board_HW = np.zeros((board_size, board_size), dtype=np.int8)
    board_HW[3, 3] = 1 # BLACK
    board_HW[4, 4] = 2 # WHITE

    past_board = np.zeros((board_size, board_size), dtype=np.int8)
    past_board[3, 3] = 1 # Only BLACK was present in the past move

    history_boards = [past_board, None, None, None, None, None, None]
    legal_actions = [0, 1, 2, 81]

    # Evaluate unbatched
    policy_s, value_s = evaluator_single.evaluate(board_HW, 1, legal_actions, history_boards)
    assert isinstance(policy_s, dict)
    assert isinstance(value_s, float)

    # Evaluate batched
    results = evaluator_batch.evaluate_batch([(board_HW, 1, legal_actions, history_boards)])
    policy_b, value_b = results[0]

    # Verify unbatched and batched match exactly
    for k in policy_s:
        assert np.isclose(policy_s[k], policy_b[k], atol=1e-5)
    assert np.isclose(value_s, value_b, atol=1e-5)

    evaluator_batch.close()


def test_vectorized_pool_swapping_gameplay_and_dataset(dummy_checkpoint: Path, tmp_path: Path):
    """E2E verification of dynamic pool-swapping play_vectorized_games with 18 channels."""
    board_size = 9
    evaluator = BatchedMLXEvaluator(dummy_checkpoint, board_size, batch_size=4, timeout_ms=2.0, channels=32, n_blocks=2, value_hidden=16, in_channels=18)

    # Play 6 total games using a pool of max_active_games=3
    black_evals = [evaluator] * 6
    white_evals = [evaluator] * 6

    records = play_vectorized_games(
        black_evaluators=black_evals,
        white_evaluators=white_evals,
        board_size=board_size,
        max_moves=10, # Keep it extremely short for fast testing
        seed=42,
        n_simulations=4, # Small simulation count for fast test
        c_puct=1.5,
        dirichlet_alpha=0.0,
        max_active_games=3,
    )

    assert len(records) == 6
    for r in records:
        assert r.board_size == board_size
        assert len(r.boards) == r.num_moves
        assert len(r.mcts_policies) == r.num_moves
        assert len(r.moves) == r.num_moves

    # Save to NPZ and verify roundtrip in GoDataset with in_channels=18
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    for idx, r in enumerate(records):
        npz_path = dataset_dir / f"game_{idx:04d}.npz"
        save_game_data(r, npz_path)
        assert npz_path.exists()

    # Load via GoDataset
    ds = GoDataset(dataset_dir, board_size=board_size, in_channels=18)
    assert len(ds) == sum(r.num_moves for r in records)

    # Inspect first batch of samples
    batches = list(ds.iter_batches(4, augment=False))
    assert len(batches) > 0
    batch = batches[0]
    
    # Assert 18 channel shape contract
    assert batch["board_BHWC"].shape == (4, board_size, board_size, 18)
    assert batch["mask_BHW"].shape == (4, board_size, board_size)
    assert batch["mcts_policy_BA"].shape == (4, board_size * board_size + 1)
    
    evaluator.close()
