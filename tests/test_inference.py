"""Phase 5c — test MLXEvaluator and BatchedMLXEvaluator.

Tests that single and batched evaluators behave identically on random board positions,
and that BatchedMLXEvaluator is robust to concurrent multi-threaded execution from 8 threads.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.inference import MLXEvaluator
from autogo_mlx.model import SizeInvariantGoResNet


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Fixture to create a random SizeInvariantGoResNet model and save its weights."""
    mx.random.seed(42)
    model = SizeInvariantGoResNet(channels=128, n_blocks=10, value_hidden=64)
    checkpoint_path = tmp_path / "dummy_model.safetensors"
    model.save_weights(str(checkpoint_path))
    return checkpoint_path


def test_single_and_batched_inference_parity(dummy_checkpoint: Path) -> None:
    board_size = 9
    n_positions = 50
    rng = np.random.default_rng(42)

    # 1. Instantiate the unbatched and batched evaluators
    single_eval = MLXEvaluator(dummy_checkpoint, board_size)
    batched_eval = BatchedMLXEvaluator(
        dummy_checkpoint, board_size, batch_size=16, timeout_ms=2.0
    )

    try:
        # 2. Generate random positions and legal move subsets
        positions = []
        for _ in range(n_positions):
            board_HW = rng.integers(0, 3, size=(board_size, board_size)).astype(np.int8)
            to_play = int(rng.choice([1, 2]))

            # Generate random subset of legal moves (always include pass)
            pass_idx = board_size * board_size
            n_legal = rng.integers(1, pass_idx)
            legal_actions = list(rng.choice(pass_idx, size=n_legal, replace=False))
            legal_actions.append(pass_idx)

            positions.append((board_HW, to_play, legal_actions))

        # 3. Evaluate using the unbatched evaluator (sequentially)
        single_results = []
        for board_HW, to_play, legal_actions in positions:
            policy, value = single_eval.evaluate(board_HW, to_play, legal_actions)
            single_results.append((policy, value))

        # 4. Evaluate using the batched evaluator concurrently from 8 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(batched_eval.evaluate, board_HW, to_play, legal_actions)
                for board_HW, to_play, legal_actions in positions
            ]
            batched_results = [f.result() for f in futures]

        # 5. Assert parity (results must match within 1e-5)
        for idx, ((single_p, single_v), (batched_p, batched_v)) in enumerate(
            zip(single_results, batched_results)
        ):
            assert abs(single_v - batched_v) < 1e-5, (
                f"Value mismatch at index {idx}: single={single_v:.6f}, batched={batched_v:.6f}"
            )

            assert set(single_p.keys()) == set(batched_p.keys()), (
                f"Policy actions set mismatch at index {idx}"
            )

            for action in single_p:
                sp = single_p[action]
                bp = batched_p[action]
                assert abs(sp - bp) < 1e-5, (
                    f"Policy prob mismatch at index {idx}, action {action}: "
                    f"single={sp:.6f}, batched={bp:.6f}"
                )
    finally:
        batched_eval.close()
