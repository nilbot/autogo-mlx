# Proposal: Calibrating Elo Anchors via Ranked Human SFT

This proposal outlines the design for sourcing human-ranked Go games, parsing them, and training supervised models to act as real-world calibrated Elo anchors (500, 1500, 2200, and 2800+ Elo) on both $9 \times 9$ and $19 \times 19$ boards. This replaces the heuristic "lobotomizing" approach of scaling down MCTS simulations with a data-driven, real-world mapping.

---

## 1. Supervised Fine-Tuning (SFT) Efficiency

Training a Go model on pre-recorded games is highly efficient because **it bypasses the MCTS simulation bottleneck**. 
- During SFT, search visit counts are not required at each step. The model only needs the board state $s$ and the actual move $a$ played by the human.
- The policy head is trained via standard cross-entropy loss against the target move:
  $$\mathcal{L}_{\text{policy}} = -\sum_{i} y_i \log p(a_i \mid s)$$
- The value head is trained via binary cross-entropy against the final game outcome $z \in \{0, 1\}$:
  $$\mathcal{L}_{\text{value}} = - [z \log v(s) + (1-z) \log(1 - v(s))]$$
- **Speed**: Training on a dataset of 500,000 positions (approx. 10,000 games) for 1 epoch takes approximately **10 to 15 minutes** on the Apple Silicon GPU using our current MLX training pipeline.

---

## 2. Storage & Data Scaling

### Disk Space Requirements
- **Minimal (SFT Anchors only)**: **10 GB** is sufficient.
  - Raw SGF files for 50,000 games ($9\times9$ and $19\times19$) consume only **~200 MB** of disk space.
  - Compressed parsed feature files consume **~6 GB** total.
- **Optimal (Future 19x19 RL Runs)**: **250 GB to 500 GB**.
  - Running a full 20-iteration reinforcement learning loop on $19\times19$ requires storing massive sliding-window replay buffers. At $19\times19$ size, a 50,000-game buffer can take 20–40 GB even when compressed, and multiple parallel runs will require hundreds of gigabytes of scratch space.

### Optimization Mechanics
To ensure that storage and loading do not become training bottlenecks, we implement the following:
- **Zstandard (zstd) Compression**: Decompression speeds exceed 600 MB/s, saturating GPU training throughput without CPU bottlenecking. For a detailed breakdown of the benchmarks, see the [Zstandard Compression Advantages Guide](qna/zstd-compression-advantages.md).
- **Memory-Mapped SGF Generator**: An on-the-fly python generator combined with bounded prefetching keeps the runtime memory footprint under 500 MB. For the implementation design, see the [Memory-Efficient SGF Parsing Guide](qna/memory-efficient-sgf-parsing.md).

---

## 3. Dataset Sizing & Rating Mapping

We crawl games for both $9 \times 9$ and $19 \times 19$ board sizes. For $19 \times 19$ games, we source professional games from GoGoD or KGS, and amateur games from Baduk servers.

| Target Bracket | Ranks (Kyu/Dan) | OGS/KGS Rank | Est. Game Count (9x9 / 19x19) | Est. Positions (9x9 / 19x19) |
| :---: | :---: | :---: | :---: | :---: |
| **500 Elo** | Double Digit Kyu (DDK) | 20k to 15k | 10,000 games | 500k / 2.0M |
| **1500 Elo** | Single Digit Kyu (SDK) | 8k to 4k | 15,000 games | 750k / 3.0M |
| **2200 Elo** | Low Dan / Advanced | 1d to 3d | 15,000 games | 750k / 3.0M |
| **2800+ Elo** | High Dan / Professional | 7d+ / Pro | 10,000 games | 500k / 2.0M |

---

## 4. Fully Convolutional Board Size Transfer ($9 \times 9 \rightarrow 19 \times 19$)

Our ResNet model is a **Fully Convolutional Network (FCN)**. Because convolutional weights are translation-invariant and local, the same filters trained on a $9\times9$ grid are applied directly to a $19\times19$ grid. 

> [!NOTE]
> This architecture does not require weight padding, zero-initialization of layers, or layer duplication.

The transition from $9 \times 9$ to $19 \times 19$ involves only two changes at the FFI/model boundary:
1. **Input Shape**: The input tensor shape changes from `(B, 9, 9, channels)` to `(B, 19, 19, channels)`.
2. **Policy Head Mapping**: The spatial convolution outputs a grid of size `board_size * board_size` (81 logits for $9\times9$, 361 logits for $19\times19$). The PASS move is concatenated to the end of the flat logits (producing a 362-length vector).

For the complete mathematical proof and architecture of this zero-shot transfer, see the [FCN Size-Invariant Board Transfer Mechanics QnA](qna/fcn-size-transfer-mechanics.md).

---

## 5. Calibration Verification & SFT Training Design

### 1. Training from Scratch vs. Fine-tuning
To build stable, calibrated anchors, **we train the models from scratch on each bracket's human dataset** rather than fine-tuning a pre-trained strong checkpoint.
- **Why not fine-tune?**: Fine-tuning a strong model (e.g. Iter 20) on low-rank data (e.g. 1500 Elo) creates a severe mismatch. While the policy prior might learn to predict low-rank moves and openings, the underlying MCTS search evaluations are driven by a value head that still possesses superhuman positional judgment. The search will aggressively correct and override the policy's human-like mistakes, resulting in a hybrid bot that still plays far stronger than a true 1500 Elo player.
- **Training from Scratch**: Training from scratch ensures that both the policy prior (capturing kyu/dan shape biases and openings) and the value head (evaluating win rate based on typical outcomes of that bracket) natively align to the target skill level.

### 2. Policy Head Target Accuracy
Because lower-ranked players behave more erratically, the maximum predictable cross-entropy accuracy decreases as skill level drops. We target the following convergence accuracies on validation holdouts:
- **500 Elo (DDK)**: Policy Accuracy **~25% – 30%**.
- **1500 Elo (SDK)**: Policy Accuracy **~35% – 40%**. 
- **2200 Elo (Dan)**: Policy Accuracy **~42% – 48%**.
- **2800+ Elo (Pro)**: Policy Accuracy **~50% – 55%**. 

---

## 6. Measuring Success & Calibration

### 1. Verification of the Anchor Hierarchy
To prove that our anchors are successfully trained, we will run a round-robin tournament between the four SFT models:

$$\text{Anchor}_{2800+} \rightarrow \text{Anchor}_{2200} \rightarrow \text{Anchor}_{1500} \rightarrow \text{Anchor}_{500}$$

We define success as establishing a strict transitive win-rate hierarchy, where each tier defeats the tier below it with a win rate of **$\ge 65\%$** (corresponding to an Elo gap of $\ge 100$ points).

### 2. Rating the RL Checkpoints
To measure the strength and improvement of our RL models:
1. **Starting Point**: Play a tournament of 20 games against the 4 anchors.
2. **Elo Mapping**: Calculate the model's Elo based on its win rate $W$ using the standard logistic formula:
   $$\text{Elo}_{\text{model}} = \text{Elo}_{\text{anchor}} - 400 \log_{10}\left(\frac{1}{W} - 1\right)$$
3. **Tracking Progress**: As the model trains, we repeat the evaluation. We expect a successful run to progress systematically, for example, transitioning from losing to the 1500 Elo anchor, to beating it, and eventually challenging the Dan-level anchors.

For details on how we resolve Elo compression and perform absolute grounding to OGS ranks using reference engines, see the [Offline Elo Calibration and Validation Guide](qna/offline-elo-calibration.md).

---

## 7. Historical Context: AlphaGo vs. AlphaZero

### 1. AlphaGo: SFT Limits & Transition to RL Self-Play
In the original AlphaGo Fan (2016) and AlphaGo Lee (2016) pipelines:
- **Supervised Learning (SFT)**: The policy network was trained on 30 million positions from 160,000 human professional games. It reached a professional move prediction accuracy of **57.0%**. This SFT-only agent achieved a strong amateur level (around 5-dan), defeating other Go programs but lacking the depth to challenge top human professionals.
- **RL Self-Play Transition**: To improve past human limits, AlphaGo **immediately transitioned to RL self-play**. The policy network was cloned, and RL was used to optimize the weights via policy gradient self-play matches (running for 1.28 million games). This RL policy network defeated the SFT model in **80%** of games.
- **Value Head Self-Play Sourcing**: To train the value network without human bias, AlphaGo generated a distinct dataset of 30 million positions *exclusively* from RL self-play games (preventing the model from overfitting to specific human game corridors).

---

### 2. AlphaZero: Bypassing Humans & Final Elo Brackets
AlphaGo Zero (2017) and the generalized AlphaZero (2017) discarded human data entirely, training solely via tabula rasa self-play:
- **AlphaGo Lee Level**: Bypassed in **36 hours** of self-play.
- **AlphaGo Master Level (60-0 online pro winner)**: Bypassed in **40 days** (29 million self-play games).

#### Final Elo Ratings (Zero/AlphaZero League):
To measure progress, DeepMind calibrated ratings by running round-robin matches between all historical iterations and human baselines:

```
Elo Rating
 5000 |--------------------------------------------- AlphaGo Zero (40-day: 5185 Elo)
      |                                              AlphaZero (34-hour: ~5020 Elo)
 4500 |--------------------------------- AlphaGo Master (4858 Elo)
      |
 4000 |
      |
 3500 |-------------------- AlphaGo Lee (3739 Elo) - Defeated Lee Sedol 4-1
      |
 3000 |---------- AlphaGo Fan (3144 Elo) - Defeated Fan Hui 5-0
      |
 2500 |
      |
 2000 |
      |
 1500 |------ Human SDK / Intermediate
      |
 1000 |
      |
  500 |-- Human DDK / Beginner
```

- **AlphaGo Fan (3144 Elo)**: Defeated 2-dan pro Fan Hui 5-0.
- **AlphaGo Lee (3739 Elo)**: Defeated 9-dan pro Lee Sedol 4-1.
- **AlphaGo Master (4858 Elo)**: Defeated top professionals 60-0 online.
- **AlphaGo Zero (5185 Elo)**: Defeated AlphaGo Master by 89 to 11.
- **AlphaZero (~5000+ Elo)**: The generalized architecture trained for 34 hours, defeating the 3-day AlphaGo Zero by 60 to 40.

---

## 8. Next Steps & Planning
We will execute this SFT pipeline in a dedicated session once the $9 \times 9$ RL loop completes:
1. **Crawler Script**: Write `scripts/crawl_games.py` to retrieve raw SGF records from OGS and other sources.
2. **Parser Generator**: Write `autogo_mlx/sgf_generator.py` for space-efficient, on-the-fly SGF parsing.
3. **Training Script**: Run SFT runs for the selected anchor configurations.
4. **Elo Evaluation**: Update the SKILL to use SFT models for anchor matches.
