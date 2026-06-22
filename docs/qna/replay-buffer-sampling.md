# How does replay buffer symlinking and dataset sampling work in AutoGo-MLX?

## Context
This question arose during the triage of Iteration 7 (producing `iter8`), which failed the 55% win rate evaluation gate against `iter7`. We checked whether the sliding window replay buffer utilized a weighted/staired sampling approach across iterations (e.g., 80% current, 15% previous, 5% older) or if it mixed them uniformly.

## Answer

### 1. Replay Buffer Structure (Symlinking)
At each iteration $N$ of the reinforcement learning loop, the orchestrator script ([run_iteration.sh](file:///Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/run_iteration.sh)) cleans the replay buffer directory (`experiments/001_train_from_scratch/selfplay/replay_buffer`) and symlinks all completed self-play games (each iteration generates exactly 1,500 games) from the current iteration and the past two iterations:
* **Current iteration**: `selfplay/iter{N}`
* **Previous iteration**: `selfplay/iter{N-1}`
* **Two iterations ago**: `selfplay/iter{N-2}`

To prevent filename collisions in the flat replay buffer directory, each symlinked file is prefixed with its source iteration index (e.g., `iter7_game_0000.npz`).

### 2. Dataset Loading & Shuffling
The training script ([train.py](file:///Users/nilbot/playground/autogo-mlx/experiments/001_train_from_scratch/train.py)) initializes the dataset using the [GoDataset](file:///Users/nilbot/playground/autogo-mlx/src/autogo_mlx/dataset.py#L283) class:
* `GoDataset` scans all `.npz` files in `replay_buffer`, parsing position counts per file and computing a global cumulative sum over all positions.
* When training starts, the batch generator is initialized with global shuffling enabled:
  ```python
  batch_iter = dataset.iter_batches(args.batch_size, shuffle=True, augment=True)
  ```
* Inside `iter_batches`, positions are sampled via a uniform global random permutation of the entire indices list:
  ```python
  n = self.total_positions
  order = rng.permutation(n) if shuffle else np.arange(n)
  ```

### 3. Sampling Properties & Rationale
Because each iteration contributes exactly 1,500 games, and the average game length remains stable (e.g., ~118-126 plies), the global uniform shuffle results in a balanced mixture:
* **~33.3%** of positions are sampled from the current iteration $N$.
* **~33.3%** of positions are sampled from Iteration $N-1$.
* **~33.3%** of positions are sampled from Iteration $N-2$.

#### 💡 The Rationale for Uniform Sampling (AlphaZero Design)
No "staired" weighting (e.g., 80% current, 15% previous, 5% older) is applied. In AlphaZero-style reinforcement learning:
1. **Suppressing Idiosyncrasies**: Sampling uniformly from a sliding window of historical iterations prevents the policy from over-optimizing or collapsing to the current generation's specific strategic quirks (idiosyncrasies).
2. **Value Head Regularization**: The value head must predict the expected game outcome from any state. Training it on a mixture of games played by older versions regularizes it, smoothing the value function and ensuring stable gradients.
3. **Preventing Rock-Paper-Scissors Cycles**: If the network trained exclusively or primarily on the current model's games, it could easily cycle through circular strategies. Uniform history mixture acts as a memory buffer stabilizing policy convergence.

