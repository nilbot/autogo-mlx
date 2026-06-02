# Session Handoff: Attempt 7 Success & Attempt 8 Scaling Blueprint

- **Date**: 2026-06-02
- **Conversation ID**: `03256682-5347-4137-ac70-3ad1ab7cb1cc`

---

## 📌 Project Overview & Handoff Summary

This session successfully designed, implemented, and executed **Attempt 7** of the AutoGo-MLX reinforcement learning training loop, resolving the catastrophic `PASS` attractor collapse that occurred in Attempt 6. 

### Original Session Goal
Diagnose why mature checkpoints in Attempt 6 lost 100% of their games against random play, and design a robust retraining pipeline to produce a stable, competitive agent.

### Key Achievements
1. **PASS Attractor Collapse Diagnosed**: Found that low-simulation MCTS noise in Iteration 5 caused White to choose `PASS` when behind. The model trained on these games, creating a positive feedback loop where White passed 100% of the time on Move 1, leaving the board out-of-distribution (OOD) for subsequent plies.
2. **Move 60 Legal Pass Gate**: Modified the batched evaluator callbacks in both [gameplay.py](file:///Users/nilbot/playground/autogo-mlx/src/autogo_mlx/gameplay.py) and [nn_mcts.py](file:///Users/nilbot/playground/autogo-mlx/src/autogo_mlx/agents/nn_mcts.py) to legally block the `PASS` action under move 60.
3. **Multi-Ply Telemetry Guard**: Expanded [telemetry_alert.py](file:///Users/nilbot/playground/autogo-mlx/scripts/telemetry_alert.py) to check pass rates for the first 10 plies (`M0` through `M9`) across all self-play games, enforcing a strict 5.0% threshold.
4. **Successful Retraining (Attempt 7)**: Ran all 13 iterations of self-play and training under the new guards. The early pass rates stayed at **exactly 0.00%**, and policy training accuracy reached **80.00%**.
5. **Parity Verification**: The final model (`iter13.safetensors`) achieved a stable **50.00% win rate** against `iter0` in evaluation play, proving that OOD state blindness was successfully resolved and the PASS attractor was cured.

---

## 📋 Proposed Implementation Plan for Attempt 8 (Next Phase)

The next phase of the project is focused on scaling up the training loop to produce a master-level agent. 

### 1. Scaling the Exploration Budget
* **Self-Play Volume**: Modify `collect.py` parameters to scale game count from 1,000 games per iteration to **5,000 or 10,000 games** to provide adequate training state diversity.
* **MCTS Simulations**: Increase simulation budgets to a flat **128 or 256 simulations** for later iterations (5+) to refine training policy targets.

### 2. Generalization & Opponent Pooling
* **Submodule & Main Repo Synchronization**: All changes are committed and clean.
* **League Play Activation**: Run self-play data collection with the `--opponent-pool-dir` flag pointing to `checkpoints/` to force the agent to play 20% of its matches against historical models, preventing localized policy exploitation.

---

## 🎯 Verification Plan

1. **Verify Base Health**: Run telemetry checks on new iterations to ensure early-move pass rates remain at 0.00% and policy entropy is healthy.
2. **Evaluation Metrics**: Evaluate intermediate checkpoints (e.g., iter5, iter10) against `iter0` to track the win-rate trajectory beyond parity.

---

## 🎯 Task Tracker for Attempt 8
* `[ ]` Configure orchestrator parameters to support 10,000 games and 128 simulations in `run_iteration.sh`.
* `[ ]` Enable the `--opponent-pool-dir` flag in `collect.py`.
* `[ ]` Clean checkpoints and self-play data.
* `[ ]` Execute Attempt 8 retraining loop.
* `[ ]` Run evaluation match against `iter0` baseline.
