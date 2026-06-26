# Walkthrough: Phase 2 Retraining Run and Architecture Completion

We have successfully executed the Phase 2 reinforcement learning training run using MLX on Apple Silicon. The run leveraged our native C++ MCTS integration, Playout Cap Randomization (PCR), dynamic resignation calibration, and Option A decoupled head architecture.

---

## 🚀 Key Accomplishments & Metrics

* **Model Decoupling (Option A)**: Resolved policy-value representation conflicts in the shared trunk by inserting `mx.stop_gradient` after the shared ResNet trunk and routing features to independent evaluation blocks.
* **PASS Attractor Cured**: Completely prevented early-game passing by implementing a Move 60 PASS legal gate and multi-ply telemetry checks (early pass rate stayed at **0.00%**).
* **Final Evaluation Victory**:
  - **Final Evaluation Checkpoint**: `iter21.safetensors`
  - **Predecessor Checkpoint**: `iter20.safetensors`
  - **Live Evaluation Gate Details**: 40 games, MCTS D4 ensembling enabled, search noise disabled.
  - **Evaluation Gate Score**: Model **22** wins, Opponent (iter20) **18** wins.
  - **Evaluation Gate Win Rate**: **55.0%** (Target: $\ge 55\%$, Gate Passed).

---

## 📊 Summary of Phase 2 Iteration Progress

The reinforcement learning training run completed 21 iterations of self-play and optimization:

| Stage | Policy Accuracy | Loss | Duration | Outcome vs. Predecessor / Status |
| :--- | :---: | :---: | :---: | :---: |
| **Bootstrap Iter 0** | 6.97% | 17.4180 | 552.2s | Bootstrap Initialized |
| **Iteration 1** | 10.58% | 13.9540 | 544.3s | — |
| **Iteration 2** | 12.16% | 14.0498 | 548.0s | — |
| **Iteration 3** | 13.23% | 14.7215 | 1196.0s | — |
| **Iteration 4** | 22.28% | 19.0917 | 1340.6s | — |
| **Iteration 5** | 27.64% | 22.1974 | 1494.0s | — |
| **Iteration 6** | 32.02% | 23.6618 | 1666.1s | — |
| **Iteration 7** | 32.25% | 22.9716 | 1592.1s | — |
| **Iteration 8** | 33.03% | 19.2796 | 3094.0s | — |
| **Iteration 9** | 35.09% | 15.3694 | 3027.2s | — |
| **Iteration 10** | 36.81% | 12.5665 | 2336.9s | — |
| **Iteration 11** | 39.86% | 10.1417 | 1924.1s | — |
| **Iteration 12** | 53.63% | 10.9788 | 1940.6s | — |
| **Iteration 13** | 51.44% | 12.2545 | 1852.5s | — |
| **Iteration 14** | 53.83% | 12.4771 | 1929.4s | — |
| **Iteration 15** | 56.33% | 13.0079 | 1872.7s | — |
| **Iteration 16** | 54.78% | 13.8262 | 2020.6s | — |
| **Iteration 17** | 54.20% | 14.0402 | 2023.1s | — |
| **Iteration 18** | 53.18% | 15.5311 | 2268.2s | — |
| **Iteration 19** | 57.32% | 15.7717 | 2188.8s | — |
| **Iteration 20** | 56.01% | 16.0954 | 2062.2s | — |
| **Iteration 21** | 58.49% | 15.7576 | 1835.0s | **55.0% Win Rate** vs iter20 (Success) |
