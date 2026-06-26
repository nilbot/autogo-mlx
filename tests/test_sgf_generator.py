"""Unit tests for the memory-efficient SGF dataset generator."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from autogo_mlx.sgf_generator import SGFDataset, SGFDatasetIterator


def test_sgf_dataset_generator(tmp_path: Path) -> None:
    # 1. Create a couple of mock SGF games in a temp directory
    sgf_content_1 = """(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RE[B+R]
;B[cd];W[cf];B[ec];W[eg];B[gf];W[gg];B[ff];W[ef];B[])"""

    sgf_content_2 = """(;GM[1]FF[4]CA[UTF-8]SZ[9]KM[7.5]RE[W+5.5]
;B[cd];W[cf];B[ec];W[eg];B[])"""

    (tmp_path / "game1.sgf").write_text(sgf_content_1, encoding="utf-8")
    (tmp_path / "game2.sgf").write_text(sgf_content_2, encoding="utf-8")

    # 2. Instantiate SGFDataset
    dataset = SGFDataset(tmp_path, board_size=9, in_channels=8, eps=0.1)
    assert len(dataset) == 2

    # 3. Iterate over SGF dataset batches
    # batch_size=4, shuffle=False to make assertions predictable
    batch_iter = dataset.iter_batches(batch_size=4, shuffle=False, augment=False)
    
    try:
        batches = list(batch_iter)
        # Game 1 has 9 moves, Game 2 has 5 moves -> total 14 positions.
        # With batch_size=4, we expect 3 full batches of 4, and 1 final batch of 2.
        assert len(batches) == 4
        
        # Check first batch
        b1 = batches[0]
        assert "board_BHWC" in b1
        assert "mask_BHW" in b1
        assert "mcts_policy_BA" in b1
        assert "winner_B" in b1
        assert "is_teacher_B" in b1

        # Check shapes
        assert b1["board_BHWC"].shape == (4, 9, 9, 8)
        assert b1["mask_BHW"].shape == (4, 9, 9)
        assert b1["mcts_policy_BA"].shape == (4, 82)
        assert b1["winner_B"].shape == (4,)
        assert b1["is_teacher_B"].shape == (4,)

        # Check values
        assert np.all(b1["mask_BHW"] == 1.0)
        assert np.all(b1["is_teacher_B"] == 1.0)
        
        # Verify policy distribution sums to 1.0
        policy_sums = b1["mcts_policy_BA"].sum(axis=-1)
        assert np.allclose(policy_sums, 1.0)

        # Check third channel (empty board mapping): first position should have only 1 stone on board
        # board_BHWC[0, ..., :3] is the one-hot board representation.
        # Channels: 0 = empty, 1 = self stones, 2 = opponent stones
        b1_board0 = b1["board_BHWC"][0]
        # First move is B[cd] which is (3, 2).
        # Since it is Black's turn (local=0), the board before move is completely empty
        assert np.all(b1_board0[..., 0] == 1.0)  # entire board is empty
        assert np.all(b1_board0[..., 1] == 0.0)
        assert np.all(b1_board0[..., 2] == 0.0)

        # Second position is White to play after Black played B[cd] (3, 2).
        # For White, Black is opponent. So Black's stone (3, 2) is in channel 2.
        b1_board1 = b1["board_BHWC"][1]
        assert b1_board1[3, 2, 2] == 1.0  # opponent stone at (3, 2)
        assert b1_board1[3, 2, 0] == 0.0  # not empty
        # All other coordinates are empty
        assert b1_board1[0, 0, 0] == 1.0

        # Check the last batch of size 2
        b4 = batches[-1]
        assert b4["board_BHWC"].shape == (2, 9, 9, 8)
        assert b4["winner_B"].shape == (2,)

    finally:
        batch_iter.close()
