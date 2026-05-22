"""Phase 11 — Smart Game Format (SGF) Parser Tests."""

from __future__ import annotations


import numpy as np

from autogo_mlx.sgf import parse_sgf_move, load_sgf_game


def test_parse_sgf_move() -> None:
    assert parse_sgf_move("") == (-1, -1)
    assert parse_sgf_move("tt") == (-1, -1)
    assert parse_sgf_move("ab") == (1, 0)
    assert parse_sgf_move("cd") == (3, 2)
    assert parse_sgf_move("ii") == (8, 8)


def test_load_sgf_game(tmp_path) -> None:
    # A tiny 9x9 game in SGF format
    sgf_content = """(;GM[1]FF[4]CA[UTF-8]AP[Go]SZ[9]KM[7.5]RE[B+12.5]
;B[cd];W[cf];B[ec];W[eg];B[gf];W[gg];B[ff];W[ef];B[])"""

    file_path = tmp_path / "game.sgf"
    file_path.write_text(sgf_content)

    record = load_sgf_game(file_path, target_size=9)
    assert record is not None
    assert record.board_size == 9
    assert record.num_moves == 9
    assert record.komi == 7.5
    assert record.winner == 1  # Black won
    assert record.result == "B+12.5"
    assert len(record.boards) == 9
    assert len(record.moves) == 9

    # Check moves
    assert record.moves[0] == (3, 2)  # cd
    assert record.moves[1] == (5, 2)  # cf
    assert record.moves[-1] == (-1, -1)  # pass

    # Check that board states alternate
    board0 = record.boards[0]
    assert np.all(board0 == 0)  # Empty board

    # Second state has Black at (3, 2)
    board1 = record.boards[1]
    assert board1[3, 2] == 1  # BLACK
    assert np.count_nonzero(board1) == 1

    # Third state has White at (5, 2)
    board2 = record.boards[2]
    assert board2[3, 2] == 1  # BLACK
    assert board2[5, 2] == 2  # WHITE
    assert np.count_nonzero(board2) == 2
