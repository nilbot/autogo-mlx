# Unified Memory Footprint and League Play Opponent Batching Design on Apple Silicon

## Context
During the transition to Phase 2 (Iteration 11), we enabled opponent pooling (league play), which evaluates games against historical checkpoints. We analyzed the design choice of keeping only one randomly selected historical opponent evaluator active per iteration (rather than drawing a new opponent per game) and examined the rationale behind keeping the GPU memory footprint strictly under 3 GB.

## Answer

### 1. The 16 GB Unified Memory Constraint
On Apple Silicon, CPU and GPU share a single **Unified Memory Architecture (UMA)**. This introduces specific system-level constraints:
* **GPU Memory Allocation Limit**: By default, macOS restricts the maximum memory allocatable to the GPU/Metal framework to approximately **70% of total physical RAM** (approx. 11.2 GB on a 16 GB machine).
* **Swap Paging Performance Degradation**: If concurrent processes (MCTS self-play, model training, IDE, browser, OS) exceed the physical RAM boundary, the operating system pages inactive memory to SSD swap space. Neural network inference inside vectorized MCTS is highly sensitive to memory latency. Triggering swap space paging turns nanosecond-latency RAM lookups into millisecond-latency SSD read/writes, collapsing self-play throughput by **10x to 50x**.

---

### 2. Single Historical Opponent Design Trade-Off
To prevent memory exhaustion and graph compilation bottlenecks, `collect.py` loads exactly **one** randomly selected historical model from the checkpoints pool per iteration, rather than drawing a different opponent per game.

#### The Problem with Multi-Opponent Instantiation:
If we randomly chose a different past checkpoint ($iter_0$ to $iter_N$) for every game in a batch of 1,000, we would have to instantiate a `BatchedMLXEvaluator` for each:

$$\text{Active Evaluators} = 1 \text{ (Current Model)} + N \text{ (Historical Generations)}$$

On Iteration 11 ($N=11$), keeping 12 evaluators active concurrently would:
1. Multiply VRAM consumption by loading 12 separate sets of model weights.
2. Introduce huge graph compilation delays at startup (each model must compile its computation graphs on Metal).
3. Introduce execution queue context-switching latency as the Metal Command Buffer switches between 12 different model graphs.

#### The Optimized Two-Model Solution:
The design restricts active models to exactly **two** per self-play run:

```python
# From experiments/001_train_from_scratch/collect.py
# Instantiates exactly one historical evaluator at the start of iteration self-play
historical_evaluator = None
if args.opponent_pool_dir:
    ...
    if valid_past:
        chosen_past = np.random.choice(valid_past)
        past_weights = mx.load(str(chosen_past))
        past_in_channels = past_weights["input_conv.weight"].shape[3]
        
        historical_evaluator = BatchedMLXEvaluator(
            checkpoint_path=chosen_past,
            board_size=args.board_size,
            batch_size=64,
            timeout_ms=1.0,
            in_channels=past_in_channels,
            d4_ensemble=args.d4_ensemble,
        )
```

During self-play, games are dynamically assigned to this single historical opponent for 20% of matches:

```python
# Games select either the current model or the single loaded historical evaluator
if historical_evaluator is not None:
    rng = np.random.default_rng(args.seed + game_idx)
    if rng.random() < 0.20:
        if rng.random() < 0.5:
            b_eval = historical_evaluator
        else:
            w_eval = historical_evaluator
```

This keeps the self-play loop memory footprint lightweight (under 3 GB) and preserves high batch inference throughput while still breaking self-play correlation over the course of multiple iterations as the historical opponent changes.
