import sys
from pathlib import Path
import mlx.core as mx
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autogo_mlx.agents.nn_mcts import MLXNNMCTSAgent
from autogo_mlx.agents.random import RandomAgent
from autogo_mlx.batched_inference import BatchedMLXEvaluator
from autogo_mlx.gameplay import play_game

# Define a modified MLXNNMCTSAgent that disables PASS under move 40
class HealthyNNMCTSAgent(MLXNNMCTSAgent):
    def select_move(self, board, seed=None, legal_actions=None, as_flat=False):
        # We override select_move to intercept the callback
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
            
        from autogo_mlx.cpp_bridge import MCTSConfig, MCTSTree, PASS_ACTION
        import numpy as np
        
        config = MCTSConfig()
        config.c_puct = self.c_puct
        config.dirichlet_alpha = self.dirichlet_alpha
        config.dirichlet_weight = 0.25 if self.dirichlet_alpha > 0.0 else 0.0
        config.temperature = self.temperature
        config.lambda_ = 0.0
        
        tree = MCTSTree(board, config)
        
        # Only allow PASS if move_count >= 40
        allow_pass = board.move_count() >= 40
        
        def batched_evaluator_cb(states):
            eval_inputs = []
            for state in states:
                board_HW = state.to_numpy()
                to_play = state.to_play()
                legal_flat = state.get_legal_moves_flat()
                
                # Intercept PASS
                if allow_pass:
                    legal_actions_nn = legal_flat + [self.pass_index]
                else:
                    legal_actions_nn = legal_flat
                    
                history_inputs_dynamic = state.get_history_numpy()
                eval_inputs.append((board_HW, to_play, legal_actions_nn, history_inputs_dynamic))
                
            results_nn = self.evaluator.evaluate_batch(eval_inputs)
            
            results = []
            for policy_nn, value_nn in results_nn:
                policy_cpp = {
                    (a if a < self.pass_index else PASS_ACTION): p
                    for a, p in policy_nn.items()
                }
                results.append((policy_cpp, value_nn))
            return results
            
        tree.run_simulations_batched(self.n_simulations, self.leaf_batch_size, batched_evaluator_cb)
        
        probs_cpp = tree.get_action_probabilities(self.temperature)
        n_actions = self.board_size * self.board_size + 1
        dense_policy = np.zeros(n_actions, dtype=np.float32)
        for act_idx, prob in probs_cpp.items():
            if act_idx == PASS_ACTION:
                dense_policy[-1] = prob
            else:
                dense_policy[act_idx] = prob
                
        total_p = dense_policy.sum()
        if total_p > 0:
            dense_policy /= total_p
        self.last_mcts_policy = dense_policy
        
        action_idx = tree.select_action(self.temperature)
        
        if as_flat or legal_actions is not None:
            if action_idx == PASS_ACTION:
                return self.pass_index
            return action_idx
        else:
            if action_idx == PASS_ACTION:
                return (-1, -1)
            return board.row_col(action_idx)

def play_single_game(game_idx, evaluator, opponent_evaluator, seed, progress_lock, stats):
    game_seed = seed + game_idx
    
    model_agent = HealthyNNMCTSAgent(
        evaluator=evaluator,
        n_simulations=64,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=0.1,
        leaf_batch_size=8,
    )
    
    opponent_agent = HealthyNNMCTSAgent(
        evaluator=opponent_evaluator,
        n_simulations=64,
        c_puct=1.0,
        dirichlet_alpha=0.0,
        temperature=0.1,
        leaf_batch_size=8,
    )
    
    model_plays_black = game_idx % 2 == 0
    if model_plays_black:
        black_agent = model_agent
        white_agent = opponent_agent
    else:
        black_agent = opponent_agent
        white_agent = model_agent
        
    record = play_game(
        black_agent=black_agent,
        white_agent=white_agent,
        board_size=9,
        max_moves=200,
        seed=game_seed,
    )
    
    model_won = False
    if record.winner == 1 and model_plays_black:
        model_won = True
    elif record.winner == 2 and not model_plays_black:
        model_won = True
        
    with progress_lock:
        stats["completed"] += 1
        if model_won:
            stats["model_wins"] += 1
        else:
            stats["opponent_wins"] += 1
        print(f"[{stats['completed']:03d}/40] Game {game_idx:03d} finished | winner={record.winner} (moves={record.num_moves}, result={record.result}) | Model Won: {model_won}", flush=True)

def main():
    print("🔬 EVALUATING ITER12 vs ITER0 WITH NO EARLY PASS...")
    
    checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter12.safetensors"
    opp_checkpoint_path = "/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/checkpoints/iter0.safetensors"
    
    evaluator = BatchedMLXEvaluator(
        checkpoint_path=checkpoint_path,
        board_size=9,
        batch_size=32,
        in_channels=8,
    )
    
    opp_evaluator = BatchedMLXEvaluator(
        checkpoint_path=opp_checkpoint_path,
        board_size=9,
        batch_size=32,
        in_channels=8,
    )
    
    progress_lock = threading.Lock()
    stats = {"completed": 0, "model_wins": 0, "opponent_wins": 0}
    
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(play_single_game, i, evaluator, opp_evaluator, 2000, progress_lock, stats)
            for i in range(40) # 40 games is plenty to verify strength
        ]
        for fut in as_completed(futures):
            fut.result()
            
    evaluator.close()
    opp_evaluator.close()
    
    duration = time.time() - t0
    win_rate = (stats["model_wins"] / 40) * 100
    print("\n==========================================")
    print("Evaluation Complete!")
    print(f"Total time: {duration:.1f} seconds")
    print(f"Model Wins (iter12): {stats['model_wins']} / 40 ({win_rate:.2f}%)")
    print(f"Opponent Wins (iter0): {stats['opponent_wins']} / 40 ({100 - win_rate:.2f}%)")
    print("==========================================")

if __name__ == "__main__":
    main()
