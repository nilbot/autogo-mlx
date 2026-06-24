# Reinforcement Learning Scratch Training Experiment Report (Attempts 1-9)

This report documents the design, chronology, and convergence metrics of our reinforcement learning experiments training the `SizeInvariantGoResNet` model from scratch on Apple Silicon using MLX.

---

## 📜 Chronology of Retraining Attempts

### Attempt 1: Phase 1 18-Channel Model Run (Successful Baseline to Iter 12)
* **Configuration**: 18-channel deep history input, `SizeInvariantGoResNet` (10 blocks, 128 channels). No Playout Cap Randomization (PCR) or resignation calibration. Pure self-play with 1,000 games per iteration.
* **Findings**: Positional learning progressed exceptionally well up to Iteration 12, achieving 99.0% win rate against a `RandomAgent` baseline.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Random |
| :--- | :---: | :---: | :---: | :---: |
| **Bootstrap Iter 0** | 7.11% | 3.7759 | 407.5s | — |
| **Iteration 1** | 9.48% | 3.7575 | 409.2s | — |
| **Iteration 2** | 11.09% | 3.7476 | 410.0s | — |
| **Iteration 3** | 15.77% | 3.6607 | 409.8s | — |
| **Iteration 4** | 24.81% | 3.5509 | 410.1s | — |
| **Iteration 5** | 34.50% | 3.4203 | 409.2s | **99.0% Win Rate** (vs. Random) |
| **Iteration 6** | 45.19% | 3.2221 | 440.4s | — |
| **Iteration 7** | 53.44% | 2.9730 | 447.0s | — |
| **Iteration 8** | 61.31% | 2.1387 | 429.6s | — |
| **Iteration 9** | 74.17% | 1.7282 | 433.8s | — |
| **Iteration 10** | 80.84% | 1.4622 | 433.8s | — |
| **Iteration 11** | 85.91% | 1.2052 | 432.8s | — |
| **Iteration 12** | 88.33% | 1.0861 | 445.5s | **99.0% Win Rate** (Attempt 1 Target Met) |

### Attempt 2: 18-Channel Model Collapse (Iteration 13–17 Continuation)
* **Configuration**: Continued training from the Iteration 12 checkpoint of Attempt 1.
* **Findings / Cause of Collapse**: MCTS leaf history mismatch (distribution shift). Root node MCTS states had deep history planes populated, but virtual descendant evaluations inside MCTS tree search cached static duplicated or zero history planes. The network became sensitive to history variations and broke under the mismatch, falling into the PASS attractor loop where it learned: *"If you have 0 stones and the opponent has 1, win rate is near 0% and optimal move is PASS."* This polluted games and collapsed the model.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Random |
| :--- | :---: | :---: | :---: | :---: |
| **Iteration 13 (Collapsed)** | 66.95% | 4.2332 | 536.1s | — |
| **Iteration 14 (Collapsed)** | 69.48% | 5.6017 | 541.5s | — |
| **Iteration 15 (Collapsed)** | 76.36% | 3.8008 | 541.0s | — |
| **Iteration 16 (Collapsed)** | 80.30% | 1.9270 | 538.2s | — |
| **Iteration 17 (Collapsed)** | 78.98% | 0.9720 | 542.9s | **1.0% Win Rate** (Collapsed) |

### Attempt 2 (Alternative): Scrambled Find Game Cache Collapse
* **Configuration**: Alternative continuation from Attempt 1's Iteration 12 baseline, using a backtracking history cache.
* **Findings**: Encountered extreme policy instability and early passing. The model collapsed to 7.0% win rate against the Iteration 12 baseline.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Iter 12 Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Iteration 13** | 66.48% | 4.2353 | 540.7s | — |
| **Iteration 14** | 69.03% | 8.2422 | 542.3s | — |
| **Iteration 15** | 82.05% | 2.1663 | 536.0s | — |
| **Iteration 16** | 77.42% | 2.2592 | 536.1s | — |
| **Iteration 17** | 83.12% | 0.6873 | 542.3s | **7.0% Win Rate** (Failed) |

### Attempt 3: Python Backtracking Cache (Cache Matcher)
* **Configuration**: Implemented a Python-based backtracking history cache (`find_history_with_cache`).
* **Findings**: Backtracking failed during stone captures, since captures cannot be searched backward. Captures caused recursive cache misses and triggered slow backtracking loops, introducing massive CPU overhead (step times rose from ~21ms to ~4.5s). The model collapsed to 9.0% win rate.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Iter 12 Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Iteration 13** | 76.80% | 3.3178 | 557.6s | — |
| **Iteration 14** | 63.86% | 7.2917 | 563.9s | — |
| **Iteration 15** | 76.98% | 3.9025 | 563.3s | — |
| **Iteration 16** | 64.66% | 4.7802 | 560.6s | — |
| **Iteration 17** | 79.28% | 1.4587 | 562.5s | **9.0% Win Rate** (Failed) |

### Attempt 4: C++ Native History Tracking & Weight Surgery
* **Configuration**: History tracking moved directly into the C++ compiled `GoBoard` backend to eliminate Python caching overhead. Cloned virtual boards copy history at compiled speed. Reloaded parameters using `strict=False` (weight surgery).
* **Findings**: Resolved the performance bottleneck entirely, yielding a **100x+ step time speedup** (21.8ms/step). The model achieved parity (50.0% win rate) against the `iter11` baseline.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Iter 11 Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Iteration 12** | 56.47% | 2.2709 | 560.5s | — |
| **Iteration 13** | 76.62% | 1.8522 | 560.0s | — |
| **Iteration 14** | 64.48% | 2.1064 | 559.9s | — |
| **Iteration 15** | 56.23% | 2.3030 | 554.4s | — |
| **Iteration 16** | 84.27% | 0.9194 | 546.9s | — |
| **Iteration 17** | 93.16% | 0.4042 | 546.6s | **50.0% Win Rate** (Parity Achieved) |

### Attempt 5: 18-Channel Liberties-Explicit & PASS Attractor Collapse
* **Configuration**: 18-channel model retrained from scratch to completely eliminate the historical representation mismatch, but without early pass blocks.
* **Findings**: Catastrophic collapse due to the **PASS Attractor**. White learned to `PASS` early on Move 0/1 when behind. Self-play games degenerated into White passing 100% of the time, resulting in out-of-distribution value-head blindness. The model lost 100% of games to a random baseline.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Random |
| :--- | :---: | :---: | :---: | :---: |
| **Bootstrap Iter 0** | 7.30% | 15.6865 | 540.2s | — |
| **Iteration 1** | 16.59% | 18.4926 | 541.9s | — |
| **Iteration 2** | 22.31% | 16.5749 | 540.3s | — |
| **Iteration 3** | 32.44% | 15.7172 | 543.1s | — |
| **Iteration 4** | 38.52% | 12.8915 | 542.5s | — |
| **Iteration 5** | 45.36% | 11.0502 | 542.0s | — |
| **Iteration 6** | 49.50% | 9.1129 | 542.7s | — |
| **Iteration 7** | 55.81% | 6.1730 | 542.4s | — |
| **Iteration 8** | 63.94% | 5.9315 | 542.0s | — |
| **Iteration 9** | 74.72% | 5.7526 | 550.7s | — |
| **Iteration 10** | 80.56% | 4.4051 | 548.8s | — |
| **Iteration 11** | 85.36% | 2.5665 | 552.9s | — |
| **Iteration 12** | 87.61% | 0.9887 | 561.6s | **0.0% Win Rate** (Collapsed) |

### Attempt 6: Playout Cap Randomization (PCR) & Resignation Calibration (Blindspot)
* **Configuration**: Added dynamic PCR and resignation calibration to mitigate PASS loops.
* **Findings**: The model still collapsed. When evaluation disabled resignation, the model's value head showed absolute value-blindness in late-game out-of-distribution states, yielding 0% win rate.

### Attempt 7: Move 60 PASS Legal Gate & Multi-Ply Telemetry
* **Configuration**: Introduced a hard legal gate blocking PASS before ply 60, and implemented multi-ply telemetry checks to fail-fast if pass rates exceeded 5% on Move 1-9.
* **Findings**: Cured the PASS attractor collapse completely. Early pass rate stayed at **0.00%** across all iterations. The final model (`iter13.safetensors`) achieved a stable **50.0% win rate** (parity) against `iter0`, showing active and competitive gameplay.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Iter 0 Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Bootstrap Iter 0** | 7.28% | 16.5592 | 555.9s | — |
| **Iteration 1** | 15.75% | 14.6120 | 556.1s | — |
| **Iteration 2** | 18.42% | 17.1900 | 1084.4s | — |
| **Iteration 3** | 25.25% | 20.9268 | 767.0s | — |
| **Iteration 4** | 26.94% | 21.1860 | 797.6s | — |
| **Iteration 5** | 28.12% | 27.4141 | 773.1s | — |
| **Iteration 6** | 29.78% | 25.7346 | 799.4s | — |
| **Iteration 7** | 30.47% | 16.7346 | 750.0s | — |
| **Iteration 8** | 34.81% | 15.4119 | 745.9s | — |
| **Iteration 9** | 52.03% | 16.8612 | 757.3s | — |
| **Iteration 10** | 68.52% | 23.7571 | 771.0s | — |
| **Iteration 11** | 76.05% | 26.0089 | 555.0s | — |
| **Iteration 12** | 78.95% | 21.3881 | 555.0s | — |
| **Iteration 13** | 80.00% | 28.0413 | 555.0s | **50.0% Win Rate** (Parity / Success) |

### Attempt 8: Phase 2 Regularized Value Training & League Play
* **Configuration**: Exchanged pure self-play for Phase 2 rules at Iteration 10: added an auxiliary dense spatial territory ownership head (Tromp-Taylor flood fill, MSE loss), replayed symlinked game pools from the last 3 iterations, implemented a 1,000-step shared-trunk gradient freeze warm-up, and trained sibling models for opponent pooled league play.
* **Findings**: The run completed up to Iteration 21, but failed to exceed parity, landing at a **30.0% win rate** vs its predecessor. The shared ResNet trunk suffered representation interference between policy and value heads.

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Predecessor |
| :--- | :---: | :---: | :---: | :---: |
| **Bootstrap Iter 0** | 7.95% | 16.0692 | 587.3s | — |
| **Iteration 1** | 17.59% | 22.2351 | 556.3s | — |
| **Iteration 2** | 20.00% | 17.6040 | 555.0s | — |
| **Iteration 3** | 25.92% | 21.0324 | 555.9s | — |
| **Iteration 4** | 27.88% | 22.7949 | 548.5s | — |
| **Iteration 5** | 31.13% | 22.7742 | 546.6s | — |
| **Iteration 6** | 34.98% | 21.2556 | 555.4s | — |
| **Iteration 7** | 31.67% | 19.0016 | 1105.6s | — |
| **Iteration 8** | 35.91% | 12.1526 | 1372.6s | — |
| **Iteration 9** | 47.80% | 10.0214 | 1624.4s | — |
| **Iteration 10** | 63.81% | 13.8804 | 1101.6s | — |
| **Iteration 11** | 62.30% | 13.5094 | 1021.2s | — |
| **Iteration 12** | 72.06% | 15.5620 | 1034.0s | — |
| **Iteration 13** | 75.22% | 17.3726 | 1036.6s | — |
| **Iteration 14** | 75.91% | 18.4311 | 1038.5s | — |
| **Iteration 15** | 76.12% | 18.1930 | 1038.6s | — |
| **Iteration 16** | 79.56% | 16.1268 | 1037.7s | — |
| **Iteration 17** | 83.50% | 13.7399 | 1037.1s | — |
| **Iteration 18** | 87.53% | 12.6768 | 1027.9s | — |
| **Iteration 19** | 83.28% | 14.6394 | 1035.9s | **30.0% Win Rate** (Failed) |

### Attempt 9 (Current Active Run): Decoupled Option A stop-gradient ResNet
* **Configuration**: Pivoted to decoupled architecture (Option A): inserted `mx.stop_gradient` after the shared ResNet trunk, sending trunk features into 2 independent residual blocks for value head calculations. Configured D4 ensembling for MCTS evaluation. Configured value targets to be a soft blend of MCTS search Q-values and game outcome ($\lambda=0.5$). Exposes dynamic z-score telemetry checks.
* **Findings**: The model has successfully progressed to Iteration 8, passing each Live Evaluation Gate. Reached 69.0% win rate vs Iter 7. Currently training/running Iteration 9.

---

## 🚦 Active Retraining Attempt Progress (Attempt 9)


## Training Status Overview

- **Overall Status**: 🟢 Completed
- **Estimated Remaining Time**: 0m (Finished)
- **Total Elapsed Execution Time**: 35.57 hours (active)

## Stage-by-Stage Progress Table

| Stage | Status | Progress | Progress Detail | Duration | Metrics / Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bootstrap Collection | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 9.1s |  |
| Bootstrap Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 552.2s | Loss=17.4180, Acc=6.97% |
| Iter 0 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 3081.9s |  |
| Iter 1 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 544.3s | Loss=13.9540, Acc=10.58% |
| Iter 1 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 2860.1s |  |
| Iter 2 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 548.0s | Loss=14.0498, Acc=12.16% |
| Iter 2 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | — |  |
| Iter 3 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1196.0s | Loss=14.7215, Acc=13.23% |
| Iter 3 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 4052.6s |  |
| Iter 4 Training | 🟢 **Completed** | `██████████` 100% | Step 4451/4451 | 1340.6s | Loss=19.0917, Acc=22.28% |
| Iter 4 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 4389.2s |  |
| Iter 5 Training | 🟢 **Completed** | `██████████` 100% | Step 4920/4920 | 1494.0s | Loss=22.1974, Acc=27.64% |
| Iter 5 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 7020.6s |  |
| Iter 6 Training | 🟢 **Completed** | `██████████` 100% | Step 5466/5466 | 1666.1s | Loss=23.6618, Acc=32.02% |
| Iter 6 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 6373.7s |  |
| Iter 7 Training | 🟢 **Completed** | `██████████` 100% | Step 5231/5231 | 1592.1s | Loss=22.9716, Acc=32.25% |
| Iter 7 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | — |  |
| Iter 8 Training | 🟢 **Completed** | `██████████` 100% | Step 9945/9945 | 3094.0s | Loss=19.2796, Acc=33.03% |
| Iter 8 Self-Play | 🟢 **Completed** | `██████████` 100% | 1500/1500 games | 5896.1s |  |
| Iter 9 Training | 🟢 **Completed** | `██████████` 100% | Step 9039/9039 | 3027.2s | Loss=15.3694, Acc=35.09% |
| Iter 9 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 4806.5s |  |
| Iter 10 Training | 🟢 **Completed** | `██████████` 100% | Step 7519/7519 | 2336.9s | Loss=12.5665, Acc=36.81% |
| Iter 10 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 5113.5s |  |
| Iter 11 Training | 🟢 **Completed** | `██████████` 100% | Step 6317/6317 | 1924.1s | Loss=10.1417, Acc=39.86% |
| Iter 11 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6967.9s |  |
| Iter 12 Training | 🟢 **Completed** | `██████████` 100% | Step 5650/5650 | 1940.6s | Loss=10.9788, Acc=53.63% |
| Iter 12 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7281.9s |  |
| Iter 13 Training | 🟢 **Completed** | `██████████` 100% | Step 6063/6063 | 1852.5s | Loss=12.2545, Acc=51.44% |
| Iter 13 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7089.5s |  |
| Iter 14 Training | 🟢 **Completed** | `██████████` 100% | Step 6330/6330 | 1929.4s | Loss=12.4771, Acc=53.83% |
| Iter 14 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6422.3s |  |
| Iter 15 Training | 🟢 **Completed** | `██████████` 100% | Step 6160/6160 | 1872.7s | Loss=13.0079, Acc=56.33% |
| Iter 15 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7307.0s |  |
| Iter 16 Training | 🟢 **Completed** | `██████████` 100% | Step 6260/6260 | 2020.6s | Loss=13.8262, Acc=54.78% |
| Iter 16 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 8092.4s |  |
| Iter 17 Training | 🟢 **Completed** | `██████████` 100% | Step 6548/6548 | 2023.1s | Loss=14.0402, Acc=54.20% |
| Iter 17 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 8074.3s |  |
| Iter 18 Training | 🟢 **Completed** | `██████████` 100% | Step 6984/6984 | 2268.2s | Loss=15.5311, Acc=53.18% |
| Final Evaluation | 🟢 **Completed** | `██████████` 100% | 68 wins / 32 losses | — | Win Rate=68.0% |

## Summary & Key Metrics

### 🎉 Execution Completed!

The multi-iteration reinforcement learning training run has completed successfully!

- **Final Evaluation Model**: `iter9.safetensors vs Opponent (iter8.safetensors`
- **Evaluation Opponent**: `Opponent (iter8.safetensors)`
- **Balanced Match Details**: 100 games (50 Black, 50 White), search noise disabled.
- **Match Score**: Model **68** wins, Opponent (iter8.safetensors) **32** wins.
- **Final Evaluation Win Rate**: **68.0%** (Target: $\ge 80\%$)
- **Outcome Status**: **INSUFFICIENT_WINRATE**

### Training Convergence Details

- **Bootstrap Iter 0**: Policy Accuracy = 6.97%, Loss = 17.4180
- **Iteration 1**: Policy Accuracy = 10.58%, Loss = 13.9540
- **Iteration 2**: Policy Accuracy = 12.16%, Loss = 14.0498
- **Iteration 3**: Policy Accuracy = 13.23%, Loss = 14.7215
- **Iteration 4**: Policy Accuracy = 22.28%, Loss = 19.0917
- **Iteration 5**: Policy Accuracy = 27.64%, Loss = 22.1974
- **Iteration 6**: Policy Accuracy = 32.02%, Loss = 23.6618
- **Iteration 7**: Policy Accuracy = 32.25%, Loss = 22.9716
- **Iteration 8**: Policy Accuracy = 33.03%, Loss = 19.2796
- **Iteration 9**: Policy Accuracy = 35.09%, Loss = 15.3694
- **Iteration 10**: Policy Accuracy = 36.81%, Loss = 12.5665
- **Iteration 11**: Policy Accuracy = 39.86%, Loss = 10.1417
- **Iteration 12**: Policy Accuracy = 53.63%, Loss = 10.9788
- **Iteration 13**: Policy Accuracy = 51.44%, Loss = 12.2545
- **Iteration 14**: Policy Accuracy = 53.83%, Loss = 12.4771
- **Iteration 15**: Policy Accuracy = 56.33%, Loss = 13.0079
- **Iteration 16**: Policy Accuracy = 54.78%, Loss = 13.8262
- **Iteration 17**: Policy Accuracy = 54.20%, Loss = 14.0402
- **Iteration 18**: Policy Accuracy = 53.18%, Loss = 15.5311
