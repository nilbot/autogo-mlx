# AutoGo System Overview and Design Philosophy

*Source: [evjang.com/2026/04/28/autogo.html](https://evjang.com/2026/04/28/autogo.html) and [github.com/ericjang/autogo](https://github.com/ericjang/autogo). Written 2026-05-15 as the orientation document for `autogo_mlx`, an MLX reimplementation.*

## What it actually is

AutoGo presents itself as a Go-playing AI, but Eric Jang is explicit in the README that **Go is the substrate, not the subject**. The project is a deliberately minimal AlphaGo-style codebase whose real purpose is to be a sandbox for *automating the AI researcher*: the entire training pipeline is driven through Claude Code, with the human providing taste, interpretation, and steering rather than typing the experiment commands directly. The repo ships `autoresearch` and `experiment` skills, and the README's "Infra Advice" reads less like a how-to-train-Go-models guide and more like field notes from an ML lead managing an autonomous engineering agent ("having Claude 'run the training loop by hand' and stop and remark when a given iteration was going unstable was very useful for catching unstable training early").

The technical claim of the tutorial is that in 2026 you can train a passable Go AI for around **~3,000 GPU-hours** — versus AlphaGo Lee's ~40,000 h, AlphaGo Zero's ~464,000 h, and KataGo's ~12,000 h. That's a preliminary number ("re-running to validate"), but the order-of-magnitude reduction is what frames the project: simple AlphaZero recipe, modern hardware (RTX 6000 Ada / RTX PRO 6000B), modern training tricks, agentic orchestration.

Why Go and not LLMs? Jang's argument is that Go has the same essential surface as frontier ML work — supervised perplexity-minimization for the policy, scalable system engineering for self-play, scaling-law studies — while being computationally lightweight enough to fit in a single researcher's budget. He also notes that the *system shape* of an AlphaGo-style trainer (logging, data collection, replay buffers, distributed RL, simulated eval) is the same shape as a robotics stack, just running orders of magnitude faster — so it doubles as a fast-iteration proxy for the kind of systems he was building at 1X.

## The system, end to end

The pipeline is the textbook AlphaZero loop with a deliberately conservative implementation.

**Self-play / data collection.** A C++ Go engine (`src/alpha_go/cpp/`) implements board state, legality, capture/Ko, and Chinese / Japanese / area scoring. On top of that sits a C++ MCTS tree with virtual loss, PUCT selection, and a Python-callback evaluator — so the tree lives in C++ but the neural-network forward pass is a Python function the tree calls back into. The Python side (`mcts.py`) duplicates the search logic in pure Python for the AlphaGo-with-rollouts variant (`lambda_>0` mixes a fast-rollout estimate with the value net), but the production path uses the C++ tree with `lambda_=0` (pure AlphaZero: value net only). `alpha_go.gameplay.play_game` and `uv run -m alpha_go.self_play` are the single front door for generating data — by convention, no experiment touches lower-level primitives. Games are written as NPZ shards with the schema `{boards: (N, H, W) int8, moves: (N, 2) int8, winner: (N,) int8, mcts_policy: (N, H*W+1) float, is_teacher: (N,) bool}`.

**Model.** `src/alpha_go/model.py` carries three architectures: a plain `GoTransformer`, a `MuPGoResNet` parameterized under `mup.MuReadout` for hyperparameter-zero-shot-transfer experiments, and the actual production network `SizeInvariantGoResNet` (128 channels, 10 residual blocks, ~3M params). The size-invariant trick is the one piece of cleverness that goes beyond the AlphaZero paper: instead of training a separate net per board size, all sizes are zero-padded into a common spatial canvas and a per-sample `mask_BHW` is propagated through every conv, normalization, pooling, and squeeze-excitation op — `MaskedBatchNorm2d` / `MaskedGroupNorm2d` re-zero the padded region after each op so it doesn't pollute neighbor convolutions. Spatial reductions divide by *true* board area, not tensor area. The output is a flat policy of length `H*W + 1` (the +1 is pass) plus a scalar value logit.

**Training.** The current production recipe is `experiments/2026-04-26_22-32-train-fromscratch/train.py`:

- AdamW, `lr=1e-3`, `weight_decay=5e-3`, 200-step linear warmup → cosine decay
- Batch size 128, board size 9, 10 dataloader workers, AMP via `torch.cuda.amp.GradScaler`
- Policy loss: per-sample cross-entropy between policy logits and the *dense MCTS visit distribution* from self-play, gated by `is_teacher` so that uninformative samples (e.g. forced random openings) don't contribute. Value loss: `binary_cross_entropy_with_logits` against the game-final winner, applied to all samples.
- 8-way D4 symmetry augmentation (rotation + reflection) at the batch level
- Early stop on `train_policy_acc >= 0.95` *or* a 15-minute wall-clock budget — both speak to the "Claude babysits the run" workflow
- Checkpoints are plain `torch.save({"model_state_dict": ..., "step": ..., "config": ...})`

**The orchestration loop.** `run_iteration.sh <start> <end>` is a bash driver that alternates `collect-it{N}` and `train-it{N+1}` jobs across a hand-rolled cluster: `cluster.toml` lists worker nodes; `infra/cluster.py` adds/removes them; `infra/remote_exec.py` SSHes to a chosen role (`train` or `collect`), `docker run --rm`s the `ghcr.io/<owner>/alphago-worker` image, rsyncs inputs in and outputs out (NFS is mounted on one node and only that node has the `train+collect` dual role). Bootstrap is a random-vs-random pre-collect to seed iter-0. The README is unusually emphatic that the author **wasted time on distributed job frameworks** and that "falling back to docker exec calls over SSH ended up working best and being agent-friendly" — that's the kind of detail you only learn by burning a weekend.

**Inference.** Two evaluators: `LocalNNEvaluator` (per-worker model copy) and `LocalBatchedInferenceEngine` (single GPU, N submitter threads coalesce requests into batches of ~64 with a 1ms timeout). A `RemoteNNEvaluator` exists for the multi-node gRPC path (`src/alpha_go/proto/inference.proto`), but the README is clear that local batching is preferred — the gRPC route is there mostly to enable multi-checkpoint A/B routing.

## What's honest about the project

A few things stand out as worth noting if you're considering reproducing it:

- It is **single-GPU per train job** by design. There is no `torch.distributed`, no DDP, no FSDP, no Lightning. Scaling out is done at the *workload* level (more collect workers, each producing NPZ shards) rather than at the gradient level. For an MLX port on a single Mac this is actively helpful — you're not undoing distributed plumbing.
- The current best checkpoint **doesn't fully understand life/death** because training uses Tromp-Taylor scoring, which doesn't model dead-stone removal. The README flags this as a known bug being worked on. So "AutoGo plays Go" is more accurate than "AutoGo plays Go well."
- The C++ MCTS tree is the load-bearing optimization, not the model. The Python tree in `mcts.py` works but is much slower. Anyone porting the system needs to decide early whether they're keeping the C++ tree as-is (it's already optimal and has no PyTorch dependency) or replacing it.
- The `mup` (Maximum Update Parameterization) machinery is present in the `MuPGoResNet` path but the production training loop uses `SizeInvariantGoResNet`, which is *not* MuP-parameterized. So a faithful port doesn't strictly need to reproduce muP — it can be a later phase.
- The whole thing is designed for an agent to drive. The experiment folder convention (`experiments/<datetime>-<slug>/{train.py, run_iteration.sh, report.md, figures/, data/, checkpoints/}`), the `is_teacher` flag in the data schema, the explicit "always go through `play_game` or `self_play`" rule in `CLAUDE.md` — these are not stylistic preferences, they are the affordances that let Claude reason about the system without getting lost.

## What's interesting about it

Setting aside the engineering, the framing is the part that lingers. Jang is using Go as a research-process testbed because the substrate is fast, cheap, and pedagogically transparent — and explicitly waving at "could we do this for biology, robotics, scaling-law studies?" The README riff on "perhaps we should be asking if P almost NP" lands harder once you've spent time inside the codebase: the whole system is a working demonstration that a learned value function can substitute for billions of dollars of microsimulation, and that this kind of substitution composes — value nets feed MCTS, MCTS generates better policy targets, the policy + value nets get sharper, the loop closes. AlphaGo is fifteen years old as an idea, and it still has interesting things to teach about what "research" looks like when the rate-limiting step is no longer the human researcher.

For me (Nils), the practical reason to port it to MLX is mundane: I want this loop running on my MacBook so I can poke it. The intellectual reason is that the *autoresearch* loop the project is really demonstrating is the same loop I want to run on my own questions — and the MLX port itself is a good first such question to hand to a scheduled agent.

## Sources

- [AutoGo: a Tutorial — Eric Jang's blog](https://evjang.com/2026/04/28/autogo.html)
- [ericjang/autogo on GitHub](https://github.com/ericjang/autogo)
- [Machines of Loving Grace — Dario Amodei](https://darioamodei.com/essay/machines-of-loving-grace) (quoted by Jang in the README on the "automate the researcher" thesis)
