# How does opponent pool contamination affect self-play reinforcement learning?

## Context
During the Iteration 18 training run, candidate `iter19.safetensors` failed the live evaluation gate against `iter18.safetensors` with a win rate of **45.00%** (18 wins, 22 losses). Diagnostic auditing of the logs revealed that opponent pooling (league play) had randomly selected the bootstrap random model `iter0.safetensors` as the opponent for 20% of the self-play games, contaminating the replay buffer.

---

## Answer

### 1. The "iter0 Effect" on Game competitive Dynamics
In AlphaZero-style self-play, opponent pooling (league play) is introduced to prevent the agent from overfitting to its own immediate playing style (circular strategies). However, selecting an opponent that is too weak (such as a random agent or a very early kyu-level checkpoint) damages training:
* **Game Length Collapse**: Because a strong agent (`iter18`) easily overpowers a random agent (`iter0`), games end prematurely. The average game length in our run collapsed from **136.4 plies** (in Iteration 17) to **106.2 plies**.
* **Tactical Sparsity**: Capture rates drop (from **29.9%** to **22.9%**) because uncompetitive games do not develop deep tactical battles, leaving the model with sparse learning signals for mid-game fights.

---

### 2. Value Head Distortion & Overconfidence
The value head $v(s)$ is trained to predict the final game outcome $z \in \{0, 1\}$ using binary cross-entropy:

$$\mathcal{L}_{\text{value}} = - [z \log v(s) + (1-z) \log(1 - v(s))]$$

When the agent plays 20% of its self-play games against a random opponent:
1. **Target Skewness**: The agent wins almost 100% of these games, meaning the target $z$ is highly biased toward a single winner.
2. **Out-of-Distribution (OOD) States**: The random opponent plays bizarre, self-capturing, or non-strategic moves. The value head is forced to train on highly atypical board positions.
3. **Loss of Granularity**: The value head learns to be overly optimistic (confident that it will win from any position) and loses the ability to differentiate tiny positional advantages in tight, high-level games. When evaluated against a serious model (like `iter18`), this overconfidence results in blunders.

---

### 3. Representation Mismatch & Telemetry Indicators
Opponent pool contamination can be detected offline before running evaluation gates using three key telemetry metrics:

* **Value-Policy Alignment ($A_{\text{VP}}$)**: Measures the Pearson correlation between the policy prior $p(a \mid s)$ and the search visit distribution $\pi(a \mid s)$. When the value head is distorted, search evaluations override the policy prior erratically, causing a drop in alignment:
  
  $$\text{Contamination Signature: } A_{\text{VP}} \text{ drops from } \approx 0.87 \text{ to } < 0.77$$

* **Opening Move Entropy ($H_{\text{open}}$)**: Under-training on diverse competitive games causes the policy prior to over-specialize in opening moves that easily crush a random agent. First-ply entropy collapses to a z-score of $<-4.0$ as probability mass spikes on a single coordinate (Tengen flat=40 at 9.0% probability):

  $$H_{\text{open}} = -\sum_{i} p(a_i) \log p(a_i)$$

* **Loss Metric Elevation**: Both final value loss (`Val`) and spatial territory ownership loss (`Own`) rise because territory borders and state outcomes in random games are highly chaotic.

---

### 4. Architectural Remediation
To prevent this failure mode, the opponent pool must be constrained to exclude early checkpoints that have not yet crossed a minimum skill baseline (e.g. Iteration 10):

```python
# From experiments/001_train_from_scratch/collect.py
for p in past_ckpts:
    m = re.search(r"iter(\d+)", p.name)
    if m:
        past_iter = int(m.group(1))
        # Restrict opponent pool to iteration >= 10 to avoid DDK/random model contamination (iter0-9)
        if past_iter < current_iter and past_iter >= 10:
            valid_past.append(p)
```

By enforcing `past_iter >= 10`, the self-play pipeline guarantees that all historical games are played against agents that understand group shape, liberties, and basic tactics, preserving the integrity of the value head and stabilizing the progression gate.
