"""Phase 3b — synthetic-NPZ smoke for :class:`autogo_mlx.dataset.GoDataset`.

Writes a tiny NPZ pair into ``tmp_path``, then asserts:

1. shape/dtype contract of :py:meth:`GoDataset.iter_batches`;
2. ``board_BHWC`` is one-hot per cell inside the mask, all-zero outside;
3. D4 augmentation preserves the per-sample policy probability mass;
4. ``shuffle=False, augment=False`` is bit-deterministic between runs.
"""

from __future__ import annotations

import numpy as np

from autogo_mlx.dataset import (
    BLACK,
    EMPTY,
    WHITE,
    GoDataset,
    _d4_apply,
    _d4_policy,
    _one_hot_board,
)


def _write_synthetic_npz(
    path,
    *,
    n: int,
    board_size: int,
    rng: np.random.Generator,
) -> None:
    boards = rng.integers(0, 3, size=(n, board_size, board_size), dtype=np.int8)
    moves = rng.integers(0, board_size, size=(n, 2), dtype=np.int8)
    raw = rng.random(size=(n, board_size * board_size + 1), dtype=np.float32) + 1e-3
    mcts_policy = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float32)
    winner = rng.integers(0, 2, size=n, dtype=np.int8)
    is_teacher = rng.integers(0, 2, size=n, dtype=bool)
    np.savez_compressed(
        path,
        boards=boards,
        moves=moves,
        mcts_policy=mcts_policy,
        winner=winner,
        is_teacher=is_teacher,
        num_moves=n,
        board_size=board_size,
    )


def test_iter_batches_contract_and_augmentation(tmp_path) -> None:
    bs = 9
    rng = np.random.default_rng(0)
    _write_synthetic_npz(tmp_path / "g0000.npz", n=8, board_size=bs, rng=rng)
    _write_synthetic_npz(tmp_path / "g0001.npz", n=4, board_size=bs, rng=rng)

    ds = GoDataset(tmp_path, board_size=bs)
    assert len(ds) == 12
    # Index gets cached on disk; subsequent construction must not re-scan files.
    assert (tmp_path / "index.json").exists()
    GoDataset(tmp_path, board_size=bs)

    batch_size = 6
    batches = list(
        ds.iter_batches(batch_size, augment=True, rng=np.random.default_rng(1))
    )
    assert len(batches) == 2

    batch = batches[0]
    assert batch["board_BHWC"].shape == (batch_size, bs, bs, 3)
    assert batch["board_BHWC"].dtype == np.float32
    assert batch["mask_BHW"].shape == (batch_size, bs, bs)
    assert batch["mask_BHW"].dtype == np.float32
    assert batch["mcts_policy_BA"].shape == (batch_size, bs * bs + 1)
    assert batch["mcts_policy_BA"].dtype == np.float32
    assert batch["winner_B"].shape == (batch_size,)
    assert batch["winner_B"].dtype == np.float32
    assert batch["is_teacher_B"].shape == (batch_size,)
    assert batch["is_teacher_B"].dtype == np.float32

    # One-hot contract inside the mask; padded cells must be all zero.
    bhwc = batch["board_BHWC"]
    mask = batch["mask_BHW"].astype(bool)
    in_region = bhwc[mask]
    assert np.all(in_region.sum(axis=-1) == 1.0)
    out_region = bhwc[~mask]
    if out_region.size:
        assert np.all(out_region == 0.0)

    # Probability mass preserved under D4 augmentation.
    sums = batch["mcts_policy_BA"].sum(axis=-1)
    np.testing.assert_allclose(sums, np.ones(batch_size), atol=1e-5)

    # winner / is_teacher are 0/1.
    assert np.all((batch["winner_B"] == 0.0) | (batch["winner_B"] == 1.0))
    assert np.all((batch["is_teacher_B"] == 0.0) | (batch["is_teacher_B"] == 1.0))


def test_no_shuffle_no_augment_is_deterministic(tmp_path) -> None:
    bs = 9
    rng = np.random.default_rng(7)
    _write_synthetic_npz(tmp_path / "g.npz", n=12, board_size=bs, rng=rng)
    ds = GoDataset(tmp_path, board_size=bs)
    a = next(iter(ds.iter_batches(12, shuffle=False, augment=False)))
    b = next(iter(ds.iter_batches(12, shuffle=False, augment=False)))
    for key in ("board_BHWC", "mask_BHW", "mcts_policy_BA", "winner_B", "is_teacher_B"):
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_one_hot_board_perspective() -> None:
    bs = 5
    board = np.array(
        [
            [EMPTY, BLACK, EMPTY, WHITE, EMPTY],
            [WHITE, EMPTY, BLACK, EMPTY, BLACK],
            [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
            [BLACK, WHITE, BLACK, WHITE, BLACK],
            [EMPTY, BLACK, WHITE, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    bk = _one_hot_board(board, BLACK)
    wh = _one_hot_board(board, WHITE)
    assert bk.shape == (bs, bs, 3) and wh.shape == (bs, bs, 3)
    # Self/opponent channels swap between perspectives.
    np.testing.assert_array_equal(bk[..., 1], wh[..., 2])
    np.testing.assert_array_equal(bk[..., 2], wh[..., 1])
    np.testing.assert_array_equal(bk[..., 0], wh[..., 0])
    # Each cell hits exactly one channel.
    np.testing.assert_array_equal(bk.sum(axis=-1), np.ones((bs, bs), dtype=np.float32))


def test_d4_apply_pass_invariant_and_orbit_size() -> None:
    rng = np.random.default_rng(3)
    bs = 9
    pol = rng.random((4, bs * bs + 1)).astype(np.float32)
    pol /= pol.sum(axis=-1, keepdims=True)

    seen = set()
    for sym in range(8):
        rotated = _d4_policy(pol, sym, bs)
        # Pass slot is untouched, regardless of sym.
        np.testing.assert_array_equal(rotated[:, -1], pol[:, -1])
        # Mass per row preserved.
        np.testing.assert_allclose(rotated.sum(axis=-1), pol.sum(axis=-1), atol=1e-6)
        seen.add(tuple(rotated[0].tolist()))
    # All 8 D4 elements act distinctly on a generic input (with prob ~ 1).
    assert len(seen) == 8

    board_HW = rng.integers(0, 3, size=(bs, bs), dtype=np.int8)
    # Applying every D4 element to a generic board yields 8 distinct boards.
    boards = {_d4_apply(board_HW, s).tobytes() for s in range(8)}
    assert len(boards) == 8


def test_18_channel_dataset_and_score(tmp_path) -> None:
    bs = 9
    rng = np.random.default_rng(42)
    # Write synthetic NPZ with custom scores
    _write_synthetic_npz(tmp_path / "g_hist.npz", n=10, board_size=bs, rng=rng)
    
    # Save a file with a final_score key to test the explicit loader
    data = dict(np.load(tmp_path / "g_hist.npz"))
    data["final_score"] = np.array([12.5], dtype=np.float32)
    np.savez_compressed(tmp_path / "g_hist.npz", **data)
    
    ds = GoDataset(tmp_path, board_size=bs, in_channels=18)
    assert len(ds) == 10
    
    batch_size = 4
    batches = list(ds.iter_batches(batch_size, augment=True))
    assert len(batches) == 2
    
    batch = batches[0]
    # Check shapes
    assert batch["board_BHWC"].shape == (batch_size, bs, bs, 18)
    assert batch["board_BHWC"].dtype == np.float32
    assert batch["mask_BHW"].shape == (batch_size, bs, bs)
    assert batch["final_score_B"].shape == (batch_size,)
    assert batch["final_score_B"].dtype == np.float32
    
    # Ensure final score loader successfully retrieved the 12.5 margin
    # (Since current_player BLACK gets margin=score, WHITE gets margin=-score)
    scores = batch["final_score_B"]
    assert np.all((scores == 12.5) | (scores == -12.5))

