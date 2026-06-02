# Phase 10: Multi-Iteration Reinforcement Learning Loops Report

This report summarizes the reinforcement learning progress of training `SizeInvariantGoResNet` from scratch on Apple Silicon using MLX.

## 🔄 Retraining Session Status (Fixed 18-Channel & Pool Swapping)

- **Overall Status**: ⏳ Scheduled (Awaiting Launch)
- **Current Active Stage**: None
- **Start Iteration**: 12 (healthy baseline)
- **Target Iteration**: 17
- **Monitoring Policy**: Unattended manual polling (No active background cron jobs)

---

## 🔄 Retraining Session Progress Table

| Stage | Status | Progress | Progress Detail | Duration | Metrics / Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Iter 12 Self-Play (Retrain) | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 3598.8s | Dynamic Pool Swapping (No Sloped Drop) |
| Iter 13 Training (Retrain) | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 540.7s | Loss=4.2353, Acc=66.48%, Val Loss=0.1331 |
| Iter 13 Self-Play (Retrain) | 🟡 **In Progress** | `▏░░░░░░░░░` 2% | 18/1000 games | — | Saturated active pool |
| Iter 14 Training (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | Step 0/2000 | — | — |
| Iter 14 Self-Play (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | 0/1000 games | — | — |
| Iter 15 Training (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | Step 0/2000 | — | — |
| Iter 15 Self-Play (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | 0/1000 games | — | — |
| Iter 16 Training (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | Step 0/2000 | — | — |
| Iter 16 Self-Play (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | 0/1000 games | — | — |
| Iter 17 Training (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | Step 0/2000 | — | — |
| Final Evaluation (Retrain) | ⏳ **Scheduled** | `░░░░░░░░░░` 0% | 0/100 games | — | Target: >= 80% Win Rate |

---

## 📜 Historical Progress & Collapse Diagnosis (Preserved)

The initial continued training session ran into a performance collapse starting in iteration 13 due to a deep-history representation mismatch (distribution shift between training history planes and MCTS evaluators). The collapsed statistics are preserved below for audit and analysis:

| Stage | Status | Progress | Progress Detail | Duration | Metrics / Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bootstrap Collection | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 14.1s |  |
| Bootstrap Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 407.5s | Loss=3.7759, Acc=7.11% |
| Iter 0 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7322.3s |  |
| Iter 1 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 409.2s | Loss=3.7575, Acc=9.48% |
| Iter 1 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7197.4s |  |
| Iter 2 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 410.0s | Loss=3.7476, Acc=11.09% |
| Iter 2 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7044.5s |  |
| Iter 3 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 409.8s | Loss=3.6607, Acc=15.77% |
| Iter 3 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7189.0s |  |
| Iter 4 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 410.1s | Loss=3.5509, Acc=24.81% |
| Iter 4 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7153.4s |  |
| Iter 5 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 409.2s | Loss=3.4203, Acc=34.50% |
| Iter 5 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7103.4s |  |
| Iter 6 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 440.4s | Loss=3.2221, Acc=45.19% |
| Iter 6 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7188.8s |  |
| Iter 7 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 447.0s | Loss=2.9730, Acc=53.44% |
| Iter 7 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6506.1s |  |
| Iter 8 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 429.6s | Loss=2.1387, Acc=61.31% |
| Iter 8 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6234.5s |  |
| Iter 9 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 433.8s | Loss=1.7282, Acc=74.17% |
| Iter 9 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6116.3s |  |
| Iter 10 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 433.8s | Loss=1.4622, Acc=80.84% |
| Iter 10 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6132.7s |  |
| Iter 11 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 432.8s | Loss=1.2052, Acc=85.91% |
| Iter 11 Self-Play | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 6463.7s |  |
| Iter 12 Training | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 445.5s | Loss=1.0861, Acc=88.33% |
| Iter 12 Self-Play (Collapsed) | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 5120.9s |  |
| Iter 13 Training (Collapsed) | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 536.1s | Loss=4.2332, Acc=66.95% |
| Iter 13 Self-Play (Collapsed) | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 5518.1s |  |
| Iter 14 Training (Collapsed) | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 541.5s | Loss=5.6017, Acc=69.48% |
| Iter 14 Self-Play (Collapsed) | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 5150.1s |  |
| Iter 15 Training (Collapsed) | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 541.0s | Loss=3.8008, Acc=76.36% |
| Iter 15 Self-Play (Collapsed) | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 7569.5s |  |
| Iter 16 Training (Collapsed) | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 538.2s | Loss=1.9270, Acc=80.30% |
| Iter 16 Self-Play (Collapsed) | 🟢 **Completed** | `██████████` 100% | 1000/1000 games | 10582.1s |  |
| Iter 17 Training (Collapsed) | 🟢 **Completed** | `██████████` 100% | Step 2000/2000 | 542.9s | Loss=0.9720, Acc=78.98% |
| Final Evaluation (Collapsed) | 🟢 **Completed** | `██████████` 100% | 1 wins / 99 losses | — | Win Rate=1.0% (Failed) |
