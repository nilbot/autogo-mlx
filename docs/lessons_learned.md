# AutoGo-MLX Reinforcement Learning: Lessons Learned & Research Findings

This document serves as a centralized knowledge base compiling the key technical insights, empirical findings, and engineering improvements discovered across all training attempts (Attempt 1 through 7). 

---

## 📌 Chronology of Retraining Attempts

### Attempt 1 & 2: The 18-Channel Model Collapse
* **Core Issue**: Mature checkpoints lost 100% of their games to random baselines.
* **Findings**:
  * **History Mismatch (Distribution Shift)**: The training dataset constructed deep history planes using actual sequential past boards. During live MCTS evaluation, the evaluator used a static approximation, duplicating the current board across all history planes. The network became sensitive to history variations and broke under the mismatch.
  * **Virtual Descendant Shift (Attempt 2)**: While the root node had correct history, any virtual descendant evaluated inside the MCTS search tree had all-zeros history planes. 
  * **PASS Feedback Loop**: Under the OOD history approximation, the model selected `PASS` on Move 0. Training on these polluted games amplified the behavior into a collapse.

### Attempt 3 & 4: History Alignment & Dynamic Caching
* **Core Issue**: Reconstructing history arrays during tree searches introduced massive CPU overhead.
* **Findings**:
  * **Dynamic Cache miss (Attempt 3)**: A Python-based backtracking cache (`find_history_with_cache`) failed during stone captures, since stone removals cannot search backwards. Descendants under capture nodes recursively cache-missed, executing slow loops.
  * **C++ Native Tracking (Attempt 4)**: History was moved directly into the C++ `GoBoard` backend. Cloned virtual boards copy their history vectors at the compiled level with zero overhead. This resulted in a **100x+ performance speedup** (step times dropped from ~4.5s to 21.8ms).

### Attempt 5 & 6: The 8-Channel liberties-Explicit & PASS Attractor
* **Core Issue**: Transitioning to a simpler 8-channel liberties-explicit representation still resulted in a 100% evaluation loss rate.
* **Findings**:
  * **The PASS Attractor (Behavioral Collapse)**: MCTS noise in Iteration 5 caused White to choose `PASS` when behind. The model trained on this, learning: *"If you have 0 stones and the opponent has 1, win rate is near 0% and optimal move is PASS."*
  * **Asymmetric selfplay**: White passed 100% of the time, while Black placed normal stones. These games took 150+ plies, bypassing single-move checks but completely corrupting the value heads.
  * **OOD State Blindness**: Forced to play without passing, value heads were completely blind, losing to random play.

### Attempt 7: Restricting Early PASS & Multi-Ply Telemetry
* **Resolution**: 
  * Disabled early passes legally for the first 60 plies.
  * Added multi-ply telemetry checks to `scripts/telemetry_alert.py` monitoring plies M1-M9.
* **Result**: Completed 13 iterations with **0.00% early pass rates** and **50.00% win rate (parity)**.

---

## ⚖️ Strategic Parameters & Trade-offs

### 1. Opponent Pooling / League Play
* **detrimental in Phase 1 (Iterations 0-10)**: Playing against older versions in early iterations dilutes the training signal with noisy, chaotic moves from weak checkpoints. This hinders the model from learning basic shapes and liberties.
* **Beneficial in Phase 2 (Iterations 10+)**: Once a stable baseline is established (policy accuracy > 75%), opponent pooling introduces variance, regularizes the value head, and prevents the policy from converging on narrow rock-paper-scissors dynamics.

### 2. GPU Saturation via Dynamic Pool Refilling (Pool Swapping)
* **Problem**: In batched self-play, standard division of games into chunks results in a "sloped drop" in batch size as games finish, causing severe GPU under-saturation.
* **Solution**: Implementing a dynamic slot pool in `play_vectorized_games` (`gameplay.py`) that refills finished slots immediately keeps the active batch size saturated at exactly 64 at all times. For memory-efficient SGF parsing architectures, see [Memory-Efficient SGF Parsing](qna/memory-efficient-sgf-parsing.md).

### 3. Liberties-Explicit representation (8-Channel vs 18-Channel)
* **8-Channel**: (Empty, Self, Opponent, 1-Liberty, 2-Liberties, 3-Liberties, 4+-Liberties, Ko).
  * *Advantage*: Very simple, requires no temporal history sequences, and is highly robust against temporal distribution shifts.
* **18-Channel**: (8 player history planes, 8 opponent history planes, player-to-move, Ko).
  * *Advantage*: Matches the standard AlphaGo Zero architecture, allowing the model to capture deep situational trends, but demands precise state history tracking.

---

## 🔗 Related Resources & Design Guides

- **Human Ranked SFT Proposal**: [Human Ranked SFT Proposal](human-ranked-sft-proposal.md)
- **Technical Guides & QnA**:
  - [FCN Size-Invariant Board Transfer Mechanics](qna/fcn-size-transfer-mechanics.md)
  - [Memory-Efficient SGF Parsing Dynamic Generator](qna/memory-efficient-sgf-parsing.md)
  - [Zstd Compression Advantages for Replay Buffers](qna/zstd-compression-advantages.md)
  - [Offline Elo Calibration and Validation Mathematics](qna/offline-elo-calibration.md)
  - [Replay Buffer Symlinking and Dataset Sampling](qna/replay-buffer-sampling.md)
  - [Preventing RL Collapse with PCR](qna/preventing-rl-collapse-with-pcr.md)
  - [Unified Memory and League Play Design](qna/unified-memory-and-league-play-design.md)
  - [Opponent Selection Pool Contamination (The iter0 Effect)](qna/opponent-pool-contamination.md)

