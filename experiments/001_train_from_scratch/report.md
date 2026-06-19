# Phase 10: Multi-Iteration Reinforcement Learning Loops Report

This report summarizes the reinforcement learning progress of training `SizeInvariantGoResNet` from scratch on Apple Silicon using MLX. It is automatically updated by the background monitor cron task.

## Training Status Overview

- **Overall Status**: 🟢 Completed
- **Estimated Remaining Time**: 0m (Finished)
- **Total Elapsed Execution Time**: 249.08 hours (active)

## Stage-by-Stage Progress Table

| Stage | Status | Progress | Progress Detail | Duration | Metrics / Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bootstrap Collection | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 8.2s |  |
| Bootstrap Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 587.3s | Loss=16.0692, Acc=7.95% |
| Iter 0 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 25957.0s |  |
| Iter 1 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 556.3s | Loss=22.2351, Acc=17.59% |
| Iter 1 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 20496.3s |  |
| Iter 2 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 555.0s | Loss=17.6040, Acc=20.00% |
| Iter 2 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 21685.7s |  |
| Iter 3 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 555.9s | Loss=21.0324, Acc=25.92% |
| Iter 3 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 21944.5s |  |
| Iter 4 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 548.5s | Loss=22.7949, Acc=27.88% |
| Iter 4 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 44362.0s |  |
| Iter 5 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 546.6s | Loss=22.7742, Acc=31.13% |
| Iter 5 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 57541.0s |  |
| Iter 6 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 555.4s | Loss=21.2556, Acc=34.98% |
| Iter 6 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 47731.3s |  |
| Iter 7 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1105.6s | Loss=19.0016, Acc=31.67% |
| Iter 7 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 49896.1s |  |
| Iter 8 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1372.6s | Loss=12.1526, Acc=35.91% |
| Iter 8 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 52887.7s |  |
| Iter 9 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1624.4s | Loss=10.0214, Acc=47.80% |
| Iter 9 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 55295.2s |  |
| Iter 10 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1101.6s | Loss=13.8804, Acc=63.81% |
| Iter 10 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | — |  |
| Iter 11 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1021.2s | Loss=13.5094, Acc=62.30% |
| Iter 11 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | — |  |
| Iter 12 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1034.0s | Loss=15.5620, Acc=72.06% |
| Iter 12 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 54707.4s |  |
| Iter 13 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1036.6s | Loss=17.3726, Acc=75.22% |
| Iter 13 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 58435.4s |  |
| Iter 14 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1038.5s | Loss=18.4311, Acc=75.91% |
| Iter 14 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 56741.8s |  |
| Iter 15 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1038.6s | Loss=18.1930, Acc=76.12% |
| Iter 15 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 49858.9s |  |
| Iter 16 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1037.7s | Loss=16.1268, Acc=79.56% |
| Iter 16 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 50948.9s |  |
| Iter 17 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1037.1s | Loss=13.7399, Acc=83.50% |
| Iter 17 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 51123.1s |  |
| Iter 18 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1027.9s | Loss=12.6768, Acc=87.53% |
| Iter 18 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 58171.7s |  |
| Iter 19 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1035.9s | Loss=14.6394, Acc=83.28% |
| Iter 19 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 50021.8s |  |
| Iter 20 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1024.9s | Loss=15.5687, Acc=84.02% |
| Iter 20 Self-Play | 🟢 **Completed** | `██████████` 100% | 10000/10000 games | 48378.6s |  |
| Iter 21 Training | 🟢 **Completed** | `██████████` 100% | Step 4000/4000 | 1040.0s | Loss=15.2823, Acc=82.45% |
| Final Evaluation | 🟢 **Completed** | `██████████` 100% | 47 wins / 53 losses | — | Win Rate=47.0% |

## Summary & Key Metrics

### 🎉 Execution Completed!

The multi-iteration reinforcement learning training run has completed successfully!

- **Final Evaluation Model**: `iter21.safetensors vs Opponent (iter11.safetensors`
- **Evaluation Opponent**: `Opponent (iter11.safetensors)`
- **Balanced Match Details**: 100 games (50 Black, 50 White), search noise disabled.
- **Match Score**: Model **47** wins, Opponent (iter11.safetensors) **53** wins.
- **Final Evaluation Win Rate**: **47.0%** (Target: $\ge 80\%$)
- **Outcome Status**: **INSUFFICIENT_WINRATE**

### Training Convergence Details

- **Bootstrap Iter 0**: Policy Accuracy = 7.95%, Loss = 16.0692
- **Iteration 1**: Policy Accuracy = 17.59%, Loss = 22.2351
- **Iteration 2**: Policy Accuracy = 20.00%, Loss = 17.6040
- **Iteration 3**: Policy Accuracy = 25.92%, Loss = 21.0324
- **Iteration 4**: Policy Accuracy = 27.88%, Loss = 22.7949
- **Iteration 5**: Policy Accuracy = 31.13%, Loss = 22.7742
- **Iteration 6**: Policy Accuracy = 34.98%, Loss = 21.2556
- **Iteration 7**: Policy Accuracy = 31.67%, Loss = 19.0016
- **Iteration 8**: Policy Accuracy = 35.91%, Loss = 12.1526
- **Iteration 9**: Policy Accuracy = 47.80%, Loss = 10.0214
- **Iteration 10**: Policy Accuracy = 63.81%, Loss = 13.8804
- **Iteration 11**: Policy Accuracy = 62.30%, Loss = 13.5094
- **Iteration 12**: Policy Accuracy = 72.06%, Loss = 15.5620
- **Iteration 13**: Policy Accuracy = 75.22%, Loss = 17.3726
- **Iteration 14**: Policy Accuracy = 75.91%, Loss = 18.4311
- **Iteration 15**: Policy Accuracy = 76.12%, Loss = 18.1930
- **Iteration 16**: Policy Accuracy = 79.56%, Loss = 16.1268
- **Iteration 17**: Policy Accuracy = 83.50%, Loss = 13.7399
- **Iteration 18**: Policy Accuracy = 87.53%, Loss = 12.6768
- **Iteration 19**: Policy Accuracy = 83.28%, Loss = 14.6394
- **Iteration 20**: Policy Accuracy = 84.02%, Loss = 15.5687
- **Iteration 21**: Policy Accuracy = 82.45%, Loss = 15.2823
