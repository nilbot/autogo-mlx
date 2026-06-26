# Phase 2 Reinforcement Learning Retraining History (A01-A10)

This document details the chronological record of our reinforcement learning training attempts. It maps our findings, failures, collapses, and key architectural pivots.

---

## 📜 Retraining Chronology

### A01: Attempt 1 — 18-Channel Model Baseline (Bootstrap to Iter 12)
* **Configuration**: 18-channel deep history input, `SizeInvariantGoResNet` (10 blocks, 128 channels). No Playout Cap Randomization (PCR) or resignation calibration. Pure self-play with 1,000 games per iteration.
* **Progress**:
  - **Bootstrap Iter 0**: Policy Accuracy = 7.11%, Loss = 3.7759
  - **Iteration 5**: Policy Accuracy = 34.50%, Loss = 3.4203 (99.0% win rate vs Random)
  - **Iteration 12**: Policy Accuracy = 88.33%, Loss = 1.0861 (Target baseline met)
* **Findings**: Positional learning progressed well. However, this model lacked regularized value heads and early passing blocks, making it highly vulnerable to representation drift.

---

### A02: Attempt 2 — 18-Channel Model Collapse (Iter 13–17 continuation)
* **Configuration**: Continued training from the Attempt 1 Iteration 12 checkpoint.
* **Outcome**: **Model Collapse** (1.0% win rate vs Random at Iteration 17).
* **Root Cause**: *MCTS leaf history mismatch*. Root MCTS search nodes had complete history planes populated. However, virtual descendant evaluations inside the C++ MCTS tree search cached static duplicated or zero history planes. The network became extremely sensitive to history variations and fell into the **PASS Attractor** loop: *"If you have 0 stones and the opponent has 1, win rate is near 0% and optimal move is PASS."* This polluted games and collapsed policy/value heads.
* **References**: See [Unified Memory and League Play Design](../qna/unified-memory-and-league-play-design.md) for history plane details.

---

### A03: Attempt 2 (Alternative) — Scrambled Find Game Cache Collapse
* **Configuration**: Continuation from Attempt 1's Iteration 12 baseline, using a backtracking history cache.
* **Outcome**: **Failed** (7.0% win rate vs Iter 12 baseline at Iteration 17).
* **Findings**: Extreme policy instability and early passing. The model was unable to resolve state sequences correctly, leading to early resignations and pass loops.

---

### A04: Attempt 3 — Python Backtracking Cache (Cache Matcher)
* **Configuration**: Implemented a Python-based backtracking history cache (`find_history_with_cache`).
* **Outcome**: **Failed** (9.0% win rate vs Iter 12 baseline).
* **Findings**: Extreme CPU performance bottleneck. Backtracking failed during stone captures because captures cannot be searched backward deterministically. Captures caused recursive cache misses and triggered slow backtracking loops. CPU step times rose from ~21ms to ~4.5s (a **200x slowdown**), making training computationally infeasible.
* **References**: See [Memory-Efficient SGF Parsing](../qna/memory-efficient-sgf-parsing.md) for parsing and sequence structures.

---

### A05: Attempt 4 — C++ Native History Tracking & Weight Surgery
* **Configuration**: History tracking moved directly into the C++ compiled `GoBoard` backend. Cloned virtual boards copy history at compiled speed. Reloaded parameters using `strict=False` (weight surgery).
* **Outcome**: **Parity Achieved** (50.0% win rate against Iteration 11 baseline).
* **Findings**: Resolved the performance bottleneck entirely, yielding a **100x+ step time speedup** (21.8ms/step). Weight surgery successfully restored the model to active gameplay.

---

### A06: Attempt 5 — 18-Channel Liberties-Explicit & PASS Attractor Collapse
* **Configuration**: 18-channel model retrained from scratch to eliminate historical representation mismatch, but without early pass blocks.
* **Outcome**: **Catastrophic Collapse** (0.0% win rate vs Random).
* **Root Cause**: The **PASS Attractor**. White learned to `PASS` early on Move 0/1 when behind by komi. Self-play games degenerated into White passing 100% of the time, resulting in out-of-distribution value-head blindness.
* **References**: See [Preventing RL Collapse with PCR](../qna/preventing-rl-collapse-with-pcr.md) for the mathematics of pass loops.

---

### A07: Attempt 6 — Playout Cap Randomization (PCR) & Resignation Calibration (Blindspot)
* **Configuration**: Added dynamic PCR and resignation calibration to mitigate PASS loops.
* **Outcome**: **Failed** (0.0% win rate).
* **Findings**: When evaluation disabled resignation, the model's value head showed absolute value-blindness in late-game out-of-distribution states.
* **References**: See [Resignation Calibration Mechanics](../qna/resignation-calibration-mechanics.md) and [Preventing RL Collapse with PCR](../qna/preventing-rl-collapse-with-pcr.md).

---

### A08: Attempt 7 — Move 60 PASS Legal Gate & Multi-Ply Telemetry
* **Configuration**: Introduced a hard legal gate blocking PASS before ply 60, and implemented multi-ply telemetry checks to fail-fast if pass rates exceeded 5% on Move 1-9.
* **Outcome**: **Success** (50.0% win rate vs Iter 0, competitive gameplay).
* **Findings**: Cured the PASS attractor collapse completely. Early pass rate stayed at **0.00%** across all iterations.

---

### A09: Attempt 8 — Phase 2 Regularized Value Training & League Play
* **Configuration**: Exchanged pure self-play for Phase 2 rules at Iteration 10: added an auxiliary dense spatial territory ownership head (Tromp-Taylor flood fill, MSE loss), replayed symlinked game pools from the last 3 iterations, implemented a 1,000-step shared-trunk gradient freeze warm-up, and trained sibling models for opponent pooled league play.
* **Outcome**: **Failed** (30.0% win rate vs predecessor).
* **Findings**: The shared ResNet trunk suffered severe *representation interference* between the policy head and the value/ownership heads. The shared parameters could not simultaneously represent local tactical patterns (policy) and global board state statistics (value and territory).
* **References**: See [Opponent Pool Contamination](../qna/opponent-pool-contamination.md) for league play mechanics.

---

### A10: Attempt 9 — Decoupled stop-gradient ResNet & Phase 2 Completion
* **Configuration**: Decoupled architecture (Option A): inserted `mx.stop_gradient` after the shared ResNet trunk, routing trunk features into 2 independent residual blocks for value head calculations to prevent policy-value gradient interference. Enabled D4 ensembling for MCTS evaluation. Value targets set to a soft blend of MCTS search Q-values and game outcome ($\lambda=0.5$). Exposes dynamic z-score telemetry checks and restricted opponent pool to avoid early-checkpoint value contamination.
* **Outcome**: **🟢 SUCCESS (Gate Passed)**
  - Completed all 21 iterations without collapse or PASS loops.
  - Final model (`iter21.safetensors`) passed the live evaluation gate vs predecessor `iter20` with a **55.0% win rate** (22 wins, 18 losses).
* **References**: See [Preventing RL Collapse with PCR](../qna/preventing-rl-collapse-with-pcr.md) and [FCN Size Transfer Mechanics](../qna/fcn-size-transfer-mechanics.md).
