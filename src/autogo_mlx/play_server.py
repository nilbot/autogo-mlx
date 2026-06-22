import argparse
import sys
import uuid
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import mlx.core as mx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure we can import from autogo_mlx
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from autogo_mlx.cpp_bridge import GoBoard, MCTSConfig, MCTSTree, PASS_ACTION
from autogo_mlx.inference import MLXEvaluator
from autogo_mlx.dataset import _one_hot_board, _compute_liberties_numpy

app = FastAPI(title="AutoGo MLX Play Server")

# Global game session registry
games: Dict[str, "MugoGame"] = {}

class MoveRequest(BaseModel):
    row: int | None = None
    col: int | None = None
    pass_move: bool = False

class NewGameRequest(BaseModel):
    model_name: str
    board_size: int = 9
    color: str = "black"
    n_simulations: int = 64

class MugoGame:
    def __init__(self, game_id: str, size: int, human_color: int, checkpoint_path: Path, n_simulations: int = 64):
        self.game_id = game_id
        self.size = size
        self.human_color = human_color  # 1 for BLACK, 2 for WHITE
        self.checkpoint_path = checkpoint_path
        self.n_simulations = n_simulations
        
        self.board = GoBoard(size)
        self.move_history: List[Tuple[int, int] | None] = []  # List of coordinates (row, col) or None for pass
        self.board_states: List[np.ndarray] = [self.board.to_numpy().copy()]
        self.to_play_history: List[int] = [self.board.to_play()]
        self.last_move: Tuple[int, int] | None = None
        
        # Load the MLX evaluator
        self.in_channels = self._detect_channels(checkpoint_path)
        self.evaluator = MLXEvaluator(
            checkpoint_path=checkpoint_path,
            board_size=size,
            in_channels=self.in_channels
        )

    def _detect_channels(self, checkpoint_path: Path) -> int:
        try:
            weights = mx.load(str(checkpoint_path))
            if "input_conv.weight" in weights:
                channels = weights["input_conv.weight"].shape[-1]
                print(f"[{self.game_id}] Auto-detected channel size = {channels} from {checkpoint_path.name}")
                return channels
        except Exception as e:
            print(f"[{self.game_id}] Warning: failed to auto-detect channels: {e}. Falling back to 8.")
        return 8

    def to_state_dict(self, message: str = "") -> dict:
        winner = self.board.get_winner()
        is_over = self.board.is_game_over()
        
        # Format result string
        result_str = None
        if is_over:
            score_margin = self.board.score()
            if winner == 1:
                result_str = f"B+{abs(score_margin):.1f}"
            elif winner == 2:
                result_str = f"W+{abs(score_margin):.1f}"
            else:
                result_str = "Draw"
                
        # Generate legal moves coordinate list
        legal_flat = self.board.get_legal_moves_flat()
        legal_coords = [self.board.row_col(idx) for idx in legal_flat]
        
        return {
            "game_id": self.game_id,
            "board": self.board.to_numpy().tolist(),
            "size": self.size,
            "to_play": self.board.to_play(),
            "last_move": self.last_move,
            "is_over": is_over,
            "result": result_str,
            "legal_moves": legal_coords,
            "human_color": self.human_color,
            "message": message,
            "score": self.board.score(),
            "last_move_was_pass": len(self.move_history) > 0 and self.move_history[-1] is None
        }

    def play_move(self, row: int | None, col: int | None) -> bool:
        if row is None or col is None:
            # Pass move
            self.board.pass_move()
            self.move_history.append(None)
            self.last_move = None
        else:
            if not self.board.is_legal(row, col):
                return False
            self.board.play(row, col)
            self.move_history.append((row, col))
            self.last_move = (row, col)
            
        self.board_states.append(self.board.to_numpy().copy())
        self.to_play_history.append(self.board.to_play())
        return True

    def undo(self) -> bool:
        # We need to undo 2 moves (bot's move + human's move) to keep it human's turn
        steps_to_undo = 2 if len(self.move_history) >= 2 else len(self.move_history)
        if steps_to_undo == 0:
            return False
            
        for _ in range(steps_to_undo):
            self.move_history.pop()
            self.board_states.pop()
            self.to_play_history.pop()
            
        # Reconstruct the board from the cached historical state
        last_board_np = self.board_states[-1]
        last_to_play = self.to_play_history[-1]
        self.board = GoBoard(self.size)
        self.board.set_from_numpy(last_board_np, last_to_play)
        
        if self.move_history:
            self.last_move = self.move_history[-1]
        else:
            self.last_move = None
        return True

    def run_mcts_analysis(self) -> dict:
        """Run MCTS search on the current position and return full evaluation statistics."""
        # 1. Config MCTS Search
        config = MCTSConfig()
        config.c_puct = 1.0
        config.dirichlet_alpha = 0.0
        config.dirichlet_weight = 0.0
        config.temperature = 1.0
        config.lambda_ = 0.0
        
        # 2. Build the search tree
        tree = MCTSTree(self.board, config)
        
        # Get legal moves flat
        legal_flat = self.board.get_legal_moves_flat()
        pass_index = self.size * self.size
        # Exclude pass moves below move 60 (to match agent rule)
        if self.board.move_count() >= 60:
            legal_actions_nn = legal_flat + [pass_index]
        else:
            legal_actions_nn = legal_flat
            
        legal_actions_set = set(legal_actions_nn)

        # Retrieve past states for 18-channel history
        history_boards = []
        if self.in_channels == 18:
            # Gather board history backward
            history_boards = [self.board_states[i] for i in range(len(self.board_states) - 2, -1, -1)]

        # Callback for C++ MCTS engine
        def mcts_evaluator_cb(state: GoBoard) -> Tuple[Dict[int, float], float]:
            state_flat_moves = state.get_legal_moves_flat()
            if state.move_count() >= 60:
                actions = state_flat_moves + [pass_index]
            else:
                actions = state_flat_moves
                
            board_HW = state.to_numpy()
            to_play = state.to_play()
            
            policy_nn, value_nn = self.evaluator.evaluate(
                board_HW, to_play, actions, history_boards
            )
            
            policy_cpp = {
                (a if a < pass_index else PASS_ACTION): p
                for a, p in policy_nn.items()
            }
            return policy_cpp, value_nn

        # 3. Simulate
        tree.run_simulations(self.n_simulations, mcts_evaluator_cb)
        
        # 4. Get root diagnostics
        root_value = float(tree.get_root_q_value())
        priors = tree.get_root_policy_priors()
        visits = tree.get_child_visit_counts()
        q_vals = tree.get_child_q_values()
        
        # Compile coordinate-mapped move recommendations
        moves_analysis = []
        for action in legal_flat:
            row, col = self.board.row_col(action)
            moves_analysis.append({
                "row": row,
                "col": col,
                "visits": visits.get(action, 0),
                "prior": priors.get(action, 0.0),
                "q_value": q_vals.get(action, 0.0)
            })
            
        return {
            "root_value": root_value,
            "moves": moves_analysis
        }

def find_all_checkpoints() -> List[Path]:
    search_paths = [
        Path("checkpoints"),
        Path("experiments/001_train_from_scratch/checkpoints")
    ]
    checkpoints = []
    for sp in search_paths:
        if sp.exists():
            for p in sp.glob("*.safetensors"):
                checkpoints.append(p.resolve())
    # Return unique sorted paths
    return sorted(list(set(checkpoints)), key=lambda x: x.name)

@app.get("/api/models")
async def get_models() -> List[str]:
    """Scan and return list of available model checkpoint filenames."""
    checkpoints = find_all_checkpoints()
    return [p.name for p in checkpoints]

@app.post("/api/new_game")
async def new_game(req: NewGameRequest) -> dict:
    """Initialize a new Go board and MCTS bot evaluator."""
    checkpoints = find_all_checkpoints()
    checkpoint_path = None
    for cp in checkpoints:
        if cp.name == req.model_name:
            checkpoint_path = cp
            break
            
    if not checkpoint_path:
        raise HTTPException(status_code=404, detail=f"Model checkpoint '{req.model_name}' not found.")
        
    game_id = str(uuid.uuid4())[:8]
    human_color = 1 if req.color == "black" else 2  # 1=BLACK, 2=WHITE
    
    game = MugoGame(
        game_id=game_id,
        size=req.board_size,
        human_color=human_color,
        checkpoint_path=checkpoint_path,
        n_simulations=req.n_simulations
    )
    games[game_id] = game
    
    # If human plays White, bot plays first move
    bot_played_move = None
    bot_analysis = None
    if human_color == 2:  # WHITE
        bot_analysis = game.run_mcts_analysis()
        # Choose action with highest MCTS visit count
        if bot_analysis["moves"]:
            best_move = max(bot_analysis["moves"], key=lambda m: (m["visits"], m["prior"]))
            game.play_move(best_move["row"], best_move["col"])
            bot_played_move = (best_move["row"], best_move["col"])
        else:
            game.play_move(None, None)  # Pass
            bot_played_move = None

    if human_color == 2:
        if bot_played_move is None:
            msg = "Game started. Bot passed."
        else:
            msg = f"Game started. Bot played at {bot_played_move[0]},{bot_played_move[1]}."
    else:
        msg = "Game initialized. Your turn."

    state = game.to_state_dict(msg)
    if bot_played_move is not None or human_color == 2:
        state["bot_move"] = bot_played_move
        state["bot_analysis"] = bot_analysis
    return state

@app.post("/api/game/{game_id}/move")
async def play_move(game_id: str, req: MoveRequest) -> dict:
    """Play human move and return bot's MCTS-guided response move."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found.")
        
    game = games[game_id]
    if game.board.is_game_over():
        return game.to_state_dict("Game is already over.")
        
    # 1. Play Human Move
    played = game.play_move(req.row, req.col)
    if not played:
        raise HTTPException(status_code=400, detail="Illegal move submitted.")
        
    # Check if game ended after human's move
    if game.board.is_game_over():
        return game.to_state_dict("Game over after move.")
        
    # 2. Play Bot Response
    bot_analysis = game.run_mcts_analysis()
    bot_played_move = None
    if bot_analysis["moves"]:
        # Find move with highest search visits, breaking ties using policy prior
        best_move = max(bot_analysis["moves"], key=lambda m: (m["visits"], m["prior"]))
        game.play_move(best_move["row"], best_move["col"])
        bot_played_move = (best_move["row"], best_move["col"])
    else:
        game.play_move(None, None)
        bot_played_move = None
        
    if game.board.is_game_over():
        msg = f"Game over. {game.to_state_dict('')['result']}"
    elif bot_played_move is None:
        msg = "Bot passed."
    else:
        msg = "Bot played move."
        
    state = game.to_state_dict(msg)
    state["bot_move"] = bot_played_move
    state["bot_analysis"] = bot_analysis
    return state

@app.post("/api/game/{game_id}/pass")
async def pass_turn(game_id: str) -> dict:
    """Submit pass move for human and play bot's response."""
    return await play_move(game_id, MoveRequest(pass_move=True))

@app.post("/api/game/{game_id}/undo")
async def undo_moves(game_id: str) -> dict:
    """Undo the last human and bot move to restore human turn."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found.")
    game = games[game_id]
    undone = game.undo()
    msg = "Undid last moves." if undone else "Cannot undo further."
    return game.to_state_dict(msg)

@app.get("/api/game/{game_id}/assist")
async def get_assist_analysis(game_id: str) -> dict:
    """Get MCTS assist details (priors, visits, Q-values) for current state."""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game session not found.")
    game = games[game_id]
    return game.run_mcts_analysis()

# Serve static files and route HTML root
static_dir = Path(__file__).resolve().parent / "web"
if static_dir.exists():
    app.mount("/web", StaticFiles(directory=str(static_dir)), name="web")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h2>Web frontend directory not found. Please verify folder setup.</h2>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Mugo MLX Local Play Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
