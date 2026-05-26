import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

# Resolve path to the compiled shared library in the build folder of the submodule
CURRENT_FILE_DIR = Path(__file__).resolve().parent
REPO_DIR = CURRENT_FILE_DIR.parent.parent
CPP_BUILD_DIR = (
    REPO_DIR / "third_party" / "autogo" / "src" / "alpha_go" / "cpp" / "build"
)

if str(CPP_BUILD_DIR) not in sys.path:
    sys.path.append(str(CPP_BUILD_DIR))

# Lazy or immediate load at import time
try:
    import alpha_go_cpp  # type: ignore[import-not-found,import-untyped]
except ImportError as e:
    raise ImportError(
        f"Could not import the compiled C++ extension 'alpha_go_cpp'. "
        f"Make sure to run the build script first: 'scripts/build_cpp.sh'. "
        f"Search path attempted: {CPP_BUILD_DIR}. "
        f"Original error: {e}"
    ) from e

if TYPE_CHECKING:
    import numpy as np

    class GoBoard:
        EMPTY: int = 0
        BLACK: int = 1
        WHITE: int = 2
        KOMI: float = 7.5

        def __init__(self, size: int = 9, komi: float = 7.5) -> None: ...
        def play(self, row: int, col: int) -> bool: ...
        def play_flat(self, index: int) -> bool: ...
        def pass_move(self) -> bool: ...
        def is_legal(self, row: int, col: int) -> bool: ...
        def is_legal_flat(self, index: int) -> bool: ...
        def get_legal_moves_flat(self) -> List[int]: ...
        def is_game_over(self) -> bool: ...
        def score(self) -> float: ...
        def get_winner(self) -> int: ...
        def size(self) -> int: ...
        def to_play(self) -> int: ...
        def move_count(self) -> int: ...
        def komi(self) -> float: ...
        def at(self, row: int, col: int) -> int: ...
        def row_col(self, flat_index: int) -> Tuple[int, int]: ...
        def copy(self) -> "GoBoard": ...
        def to_numpy(self) -> np.ndarray: ...
        def set_from_numpy(self, board_array: np.ndarray, to_play: int) -> None: ...

    class MCTSConfig:
        c_puct: float
        lambda_: float
        dirichlet_alpha: float
        dirichlet_weight: float
        temperature: float
        max_depth: int
        rollout_temperature: float
        pcr_sims: List[int]
        pcr_probs: List[float]

        def __init__(self) -> None: ...

    class MCTSTree:
        def __init__(self, root_state: GoBoard, config: MCTSConfig) -> None: ...
        def run_simulations(
            self,
            num_simulations: int,
            evaluator: Callable[[GoBoard], Tuple[Dict[int, float], float]],
        ) -> Tuple[Dict[int, float], float]: ...
        def get_action_probabilities(
            self, temperature: float = 1.0
        ) -> Dict[int, float]: ...
        def select_action(self, temperature: float = 1.0) -> int: ...
        def tree_size(self) -> int: ...
        def get_root_visit_count(self) -> int: ...
        def get_root_q_value(self) -> float: ...
        def get_root_policy_priors(self) -> Dict[int, float]: ...
        def get_child_visit_counts(self) -> Dict[int, int]: ...
        def get_child_q_values(self) -> Dict[int, float]: ...
        def get_child_first_eval_values(self) -> Dict[int, float]: ...
        def get_child_max_subtree_depths(self) -> Dict[int, int]: ...
        def run_simulations_batched(
            self,
            num_simulations: int,
            leaf_batch_size: int,
            batched_evaluator: Callable[
                [List[GoBoard]], List[Tuple[Dict[int, float], float]]
            ],
        ) -> None: ...

    class VectorizedMCTS:
        def __init__(self, root_states: List[GoBoard], config: MCTSConfig) -> None: ...
        def run_simulations(
            self,
            num_simulations: int,
            evaluator: Callable[
                [List[GoBoard]], List[Tuple[Dict[int, float], float]]
            ],
        ) -> None: ...
        def get_action_probabilities(
            self, temperature: float = 1.0
        ) -> List[Dict[int, float]]: ...
        def select_actions(self, temperature: float = 1.0) -> List[int]: ...

    PASS_ACTION: int

    def run_mcts(
        state: GoBoard,
        num_simulations: int,
        config: MCTSConfig,
        evaluator: Callable[[GoBoard], Tuple[Dict[int, float], float]],
        temperature: float = 1.0,
    ) -> Dict[int, float]: ...

else:
    # Runtime assignments
    GoBoard = alpha_go_cpp.GoBoard
    MCTSConfig = alpha_go_cpp.MCTSConfig
    MCTSTree = alpha_go_cpp.MCTSTree
    VectorizedMCTS = alpha_go_cpp.VectorizedMCTS
    PASS_ACTION = alpha_go_cpp.PASS_ACTION
    run_mcts = alpha_go_cpp.run_mcts
