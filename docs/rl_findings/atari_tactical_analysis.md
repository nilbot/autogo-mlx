# AutoGo-MLX Atari Liberty & Short Games Tactical Analysis

This report investigates two key anomalies identified in reinforcement learning experiments:
1. **Short Games (e.g., 10 moves)**: Whether "surrendering" or "premature passing" is learned or factored in.
2. **Tactical Atari Blindness (Capture/Escape)**: Why the bot plays extremely well overall but suddenly collapses when its stones run out of liberties (Qi).

---

## 1. Exhaustive Training Logs & Game Traces

Every single game played during self-play data collection is fully recorded and preserved.
* **Storage Location**: `experiments/001_train_from_scratch/selfplay/iter{X}/` (relative to the training directory root)
* **Format**: Compressed `.npz` files (e.g., `game_0000.npz`) mapped in `index.json`.
* **Trace Schema**:
  * `boards`: Exact `(num_moves, H, W)` board states before each move.
  * `moves`: Exact `(num_moves, 2)` move coordinate tuples (where `[-1, -1]` is PASS).
  * `mcts_policy`: The `(num_moves, H*W+1)` target policy distribution (MCTS visit probabilities) for every step.
  * `winner`: Absolute winner per step.

---

## 2. Analysis of the 10-Move "Surrendering" Games

By writing a custom trace analyzer across all iterations, we obtained the exact game length distributions:

| Iteration | Total Games | Mean Length | Min Length | Short Games (< 15 moves) |
| :--- | :--- | :--- | :--- | :--- |
| **iter0** | 1000 | 119.9 | 3 | 9 |
| **iter1** | 1000 | 116.8 | 3 | 5 |
| **iter2** | 1000 | 115.3 | 2 | 3 |
| **iter3** | 1000 | 117.1 | 2 | 5 |
| **iter4** | 1000 | 117.3 | 3 | 2 |
| **iter5** | 150 | 113.9 | 10 | 2 |

### Key Findings:
1. **Extremely Rare**: Games under 15 moves account for **less than 1%** of all games. The vast majority of games average a healthy **115+ moves**.
2. **Double-Pass Mechanism**: The short games are not caused by a formal "resignation" or "surrendering" move (which is not implemented in our self-play gameplay loop). Instead, they are caused by **consecutive passes**.
3. **Stochastic Blundering**: During self-play collection, both agents play with a constant **temperature of 1.0** (to encourage exploration). Occasionally, Dirichlet noise or stochastic sampling causes Black or White to select a `PASS` move stochastically. If both happen to sample a pass consecutively, the game terminates immediately and is area-scored (with White often winning due to the $7.5$ Komi).
4. **Conclusion**: The bot is *not* throwing the game due to an explicit "surrendering" state. It is simply a minor side-effect of a high-temperature stochastic policy occasionally matching two consecutive passes early in the game.

---

## 3. Deep-Dive: Tactical Atari Blindness (Qi)

To inspect why the bot suddenly collapses when its liberties are threatened, we set up a controlled **Atari Scenario** on a 9x9 board:
* **Black Stone in Atari**: at `(4, 4)`
* **White Stones surrounding it**: at `(3, 4)`, `(4, 3)`, `(4, 5)`
* **The Single Crucial Liberty**: at `(5, 4)` (flat index `49`)

We evaluated the exact policy priors, value network heads, and MCTS search visits (from 16 to 1024 simulations) using checkpoints `iter0` (random), `iter3` (mid-training), and `iter5` (fully trained):

```
+---------------------------------------+
| .   .   .   .   .   .   .   .   .     |
| .   .   .   .   .   .   .   .   .     |
| .   .   .   .   .   .   .   .   .     |
| .   .   .   .   W   .   .   .   .     |
| .   .   .   W   B   W   .   .   .     |
| .   .   .   .  [?]  .   .   .   .     |  <-- Key Liberty (5, 4)
| .   .   .   .   .   .   .   .   .     |
| .   .   .   .   .   .   .   .   .     |
| .   .   .   .   .   .   .   .   .     |
+---------------------------------------+
```

### Empirical Results from Liberty Diagnostics

#### 1. Checkpoint `iter0` (Initial Random Weights)
* **Black win prob (Value)**: `0.4675`
* **Priors**: Escape `(5, 4)` = `0.0127`, Capture `(5, 4)` = `0.0129` (uniform noise)
* **MCTS Results**:
  * Black turn: Escape move gets **0 visits** at 16, 64, and 256 sims. Best move chosen is garbage `(0, 7)`.
  * White turn: Capture move gets **0 visits across all budgets**.

#### 2. Checkpoint `iter3` (Mid Training)
* **Priors**: Escape = `0.0137`, Capture = `0.0144`
* **MCTS Results**:
  * Black turn: Escape move gets **0 visits** at 16, 64, 256 sims. At 1024 sims, it gets **3 visits** (ignored).
  * White turn: Capture move gets **0 visits** at 16, 64, 256 sims. At 1024 sims, it gets **11 visits** (ignored).

#### 3. Checkpoint `iter5` (Trained Model)
* **Scenario A (Black's turn to ESCAPE)**:
  * **Black win prob (Value)**: `0.4627`
  * **Escape prior**: `0.0145` (Still extremely low!)
  * **MCTS Search Progression**:
    * **16 sims**: visits = `0`, Q = `0.00`
    * **64 sims**: visits = `0`, Q = `0.00`
    * **256 sims**: visits = `0`, Q = `0.00`
    * **1024 sims**: visits = **`231`**, Q = **`0.5619`** (Chosen best: `(8, 1)` with `309` visits, Q = `0.3955`)
  * *Verdict*: Black **fails to escape** at standard simulation budgets (16, 64, 256). Even at 1024 simulations, although MCTS discovers that escaping is much better (`Q=0.56` vs `Q=0.39`), it is **still not chosen** because the budget was split and `(8,1)` had a massive headstart in visits due to a higher network prior.

* **Scenario B (White's turn to CAPTURE)**:
  * **White win prob (Value)**: `0.5376`
  * **Capture prior**: `0.0216` (Slightly elevated!)
  * **MCTS Search Progression**:
    * **16 sims**: visits = `15`, Q = `0.5516` (Chosen!)
    * **64 sims**: visits = `63`, Q = `0.5812` (Chosen!)
    * **256 sims**: visits = `255`, Q = `0.6013` (Chosen!)
    * **1024 sims**: visits = `610`, Q = `0.5773` (Chosen!)
  * *Verdict*: White **successfully captures** at all simulation budgets! The prior was high enough to seed a single visit, after which MCTS immediately recognized the massive reward of removing the stones.

---

## 4. Why Does Tactical Blindness Happen?

1. **Weak Prior Network (The Atari Blindspot)**: 
   The model input is strictly `board_BHWC` (3 channels: empty, self, opponent). The model does *not* receive explicit handcrafted tactical features like liberties (Qi). It has to learn to compute liberties from scratch using purely 3x3 convolutions. 
   While the network does learn to capture (prior increased to `0.0216` in `iter5`), it has **not** sufficiently learned the self-preservation of escaping Atari (prior remains low at `0.0145`).

2. **PUCT Search Bottleneck**:
   With a simulation budget of `n_simulations = 64` (standard for evaluation and play), if a move has a raw prior probability of `0.0145`, **it will never get explored**. 
   MCTS is completely dependent on the policy network's prior to guide its first visits. If the network is blind to escaping, MCTS will not spend even 1 of its 64 simulations checking that move, resulting in total blindness.

3. **No Rollout Fallback**:
   The value head `config.lambda_ = 0.0` uses a pure value network without rollouts. If the value head itself hasn't fully converged on recognizing a group in Atari as "dead" (and thus having a lower value), there is no rollout simulation to discover the tactical death downstream.

---

## 5. Architectural & Training Recommendations

To fully resolve this issue and align with premium state-of-the-art Go architectures (like KataGo), we recommend the following modifications:

### A. Handcrafted Tactical Features (Highly Recommended)
Instead of feeding only 3 absolute channels, we can expand the model input representation in `dataset.py` to include:
1. **Liberties (Qi)**: 4 binary channels representing whether a stone group has exactly 1, 2, 3, or $\ge 4$ liberties. This immediately removes "Atari blindness" by making liberties linearly readable by the first ResNet layer.
2. **Ko Points**: 1 channel marking the active Ko point.

### B. Dynamic Temperature Scheduling
During self-play data collection:
* Set `temperature = 1.0` for the first **30 moves** to ensure exploration and variety.
* Set `temperature = 0.0` (greedy selection) for the rest of the game.
* *Benefit*: This completely eliminates premature stochastic passes and garbage terminal states, resulting in cleaner value network signals.

### C. Atari / Capture Training Penalty (Heuristic)
If we want to explicitly penalize/reward capture scenarios, we can inject a auxiliary loss in `loss.py`:
* Add a secondary head that predicts the number of stones captured on the next turn, or penalize value logits directly when a friendly group is left in Atari without defense. However, adding handcrafted liberties (A) is usually sufficient to let the model learn this naturally.
