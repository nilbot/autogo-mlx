# AutoGo-MLX Retraining Strategy: Recommendations for Attempt 8

This document serves as a transition guide and strategic blueprint for picking up reinforcement learning training in a fresh session. It outlines how to scale the current robust 8-channel liberties-explicit pipeline to train a competitive Go agent.

---

## 1. Current State (Attempt 7 Success)
* **Status**: 🟢 **Completed 13 iterations successfully** without behavioral decay or collapse.
* **Core Fixes Implemented**:
  * **C++ Native History Tracking**: Complete alignment between training inputs and MCTS rollouts with zero CPU/DP cache overhead.
  * **Move 60 Legal Pass Gate**: Restricting `PASS` moves below ply 60 to prevent the model from learning a degenerate early-forfeit bypass.
  * **Multi-Ply Telemetry Checks**: `scripts/telemetry_alert.py` now scans Move 1 through 9 pass rates with a 5% fail-fast threshold.
* **Result**: **50% win rate** against `iter0` baseline. The model plays active, competitive stones on the board, but is constrained in strength by exploration size.

---

## 2. Blueprint for Attempt 8: Scaling up to Master Level
To transition from a proof-of-concept run to a high-strength Go agent, we must increase the exploration complexity and iteration depth. We recommend the following scaling parameters:

### Strategy A: Scaling the Self-Play Game Budget
* **Current**: 1,000 games per iteration.
* **Recommendation**: Scale to **5,000 or 10,000 games** per iteration.
* **Rationale**: 1,000 games (approx. 120k board states) is enough to learn basic mechanics but insufficient to generalize complex tactical structures (such as life-and-death or ko fights). Scaling to 10k games provides the dataset diversity needed for deep ResNet generalization.

### Strategy B: Escalating MCTS Simulation Counts
* **Current**: Progressive sims (`16 -> 32 -> 64`).
* **Recommendation**: Flat **64 simulations** for early iterations (0–4), scaling to **128 or 256 simulations** for mature iterations (5+).
* **Rationale**: Higher simulation budgets reduce the policy noise and allow MCTS to output extremely high-quality target distributions. Because our C++ native evaluator is highly optimized, the time-per-step penalty is minimal.

### Strategy C: Activating League Play / Opponent Pool
* **Current**: Model only plays against its current version (pure self-play).
* **Recommendation**: Enable the historical opponent pool flag in `collect.py`:
  ```bash
  --opponent-pool-dir experiments/001_train_from_scratch/checkpoints
  ```
* **Rationale**: Under pure self-play, models can develop localized tactical blindspots (rock-paper-scissors dynamics) or suffer from catastrophic forgetting. Forcing 20% of self-play games to match the active model against randomly selected historical iterations (e.g., `iter4`, `iter8`) forces robust, generalized play.

### Strategy D: Tuning the Legal Pass Threshold
* **Current**: PASS restricted below move 60.
* **Recommendation**: Keep the PASS restriction active, but monitor if we can safely lower it to **move 40** once the value head converges, or leave it at 60 as it does not affect endgame play (which typically takes 100+ plies).

---

## 3. Step-by-Step Launch Commands

When launching a new Attempt 8 training run, execute the following commands:

```bash
# 1. Clean up old checkpoints and self-play data
rm -rf experiments/001_train_from_scratch/selfplay/*
rm -rf experiments/001_train_from_scratch/checkpoints/*

# 2. Modify the orchestrator (run_iteration.sh) to include the opponent pool:
# In the collect.py invocation, add:
#   --opponent-pool-dir "${EXP_DIR}/checkpoints"

# 3. Kick off the run
# Usage: ./run_iteration.sh <start_iter> <end_iter> <in_channels>
./experiments/001_train_from_scratch/run_iteration.sh 0 20 8
```
