import numpy as np

def main():
    data = np.load("/Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/selfplay/iter5/game_0000.npz")
    boards = data["boards"]
    moves = data["moves"]
    winner = data["winner"]
    mcts_policy = data["mcts_policy"]
    
    print("Moves length:", len(moves))
    print("Boards length:", len(boards))
    print("Winner shape:", winner.shape)
    print("Winner:", winner)
    
    for i in range(5):
        print(f"\nStep {i}:")
        print("  Move:", tuple(moves[i]))
        print("  Stones count:", np.sum(boards[i] != 0))
        # Unique values on board
        print("  Board unique:", list(set(boards[i].flatten())))
        
        # Policy sum
        print("  Policy sum:", mcts_policy[i].sum())
        # Top 3 policy actions
        sorted_pol = sorted(list(enumerate(mcts_policy[i])), key=lambda x: x[1], reverse=True)[:3]
        print("  Top policy priors:", sorted_pol)

if __name__ == "__main__":
    main()
