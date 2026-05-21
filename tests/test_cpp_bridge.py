"""Phase 6c — C++ MCTS and GoBoard bridge verification."""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pytest

from mugo.cpp_bridge import GoBoard, MCTSConfig, MCTSTree, PASS_ACTION, run_mcts


def test_goboard_properties() -> None:
    board = GoBoard(9, komi=7.5)
    
    assert board.size() == 9
    assert board.komi() == 7.5
    assert board.to_play() == GoBoard.BLACK  # Should be BLACK (1)
    assert board.move_count() == 0
    assert not board.is_game_over()
    
    # Check constants
    assert GoBoard.EMPTY == 0
    assert GoBoard.BLACK == 1
    assert GoBoard.WHITE == 2


def test_goboard_scripted_game() -> None:
    board = GoBoard(9, komi=7.5)
    
    # Move 1: Black plays at (3, 3)
    assert board.is_legal(3, 3)
    success = board.play(3, 3)
    assert success
    assert board.move_count() == 1
    assert board.to_play() == GoBoard.WHITE
    assert board.at(3, 3) == GoBoard.BLACK
    
    # Move 2: White plays flat at flat_index for (3, 4)
    flat_3_4 = 3 * 9 + 4
    assert board.is_legal_flat(flat_3_4)
    success = board.play_flat(flat_3_4)
    assert success
    assert board.move_count() == 2
    assert board.to_play() == GoBoard.BLACK
    assert board.at(3, 4) == GoBoard.WHITE
    
    # Move 3: Black plays at (4, 3)
    success = board.play(4, 3)
    assert success
    assert board.move_count() == 3
    assert board.to_play() == GoBoard.WHITE
    
    # Move 4: White plays at (4, 4)
    success = board.play(4, 4)
    assert success
    assert board.move_count() == 4
    assert board.to_play() == GoBoard.BLACK
    
    # Move 5: Black passes
    success = board.pass_move()
    assert success
    assert board.move_count() == 5
    assert board.to_play() == GoBoard.WHITE
    
    # Verification of board conversions
    arr = board.to_numpy()
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (9, 9)
    assert arr.dtype == np.int8
    assert arr[3, 3] == GoBoard.BLACK
    assert arr[3, 4] == GoBoard.WHITE
    
    # Test setting state from numpy
    new_board = GoBoard(9, komi=7.5)
    new_board.set_from_numpy(arr, to_play=GoBoard.WHITE)
    assert new_board.at(3, 3) == GoBoard.BLACK
    assert new_board.at(3, 4) == GoBoard.WHITE
    assert new_board.to_play() == GoBoard.WHITE

    # Check score and winner
    score_val = board.score()
    assert isinstance(score_val, float)
    winner = board.get_winner()
    assert winner in (0, GoBoard.BLACK, GoBoard.WHITE)


def test_mcts_config_and_tree() -> None:
    board = GoBoard(9, komi=7.5)
    config = MCTSConfig()
    
    # Test reading and writing config properties
    config.c_puct = 1.5
    config.lambda_ = 0.25
    config.dirichlet_alpha = 0.03
    config.max_depth = 50
    
    assert config.c_puct == pytest.approx(1.5)
    assert config.lambda_ == pytest.approx(0.25)
    assert config.dirichlet_alpha == pytest.approx(0.03)
    assert config.max_depth == 50
    
    # Create MCTSTree
    tree = MCTSTree(board, config)
    assert tree.tree_size() == 1  # only root node initially
    assert tree.get_root_visit_count() == 0
    assert isinstance(tree.get_root_q_value(), float)
    
    # Simple Python evaluator callback
    def dummy_evaluator(state: GoBoard) -> Tuple[Dict[int, float], float]:
        legal_flat = state.get_legal_moves_flat()
        prob = 1.0 / (len(legal_flat) + 1) if legal_flat else 1.0
        policy_dict = {a: prob for a in legal_flat}
        policy_dict[PASS_ACTION] = prob
        return policy_dict, 0.5
        
    # Run a few simulations using run_simulations
    tree.run_simulations(num_simulations=10, evaluator=dummy_evaluator)
    policy_probs = tree.get_action_probabilities(temperature=1.0)
    assert isinstance(policy_probs, dict)
    assert tree.get_root_visit_count() == 10
    assert tree.tree_size() > 1

    # Test run_mcts utility function
    probs = run_mcts(board, 5, config, dummy_evaluator, 1.0)
    assert isinstance(probs, dict)
    assert len(probs) > 0

