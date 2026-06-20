#!/usr/bin/env bash
# mugo — Phase 10 Iteration Orchestrator
# Usage: ./run_iteration.sh <start_iter> <end_iter>
# Example: ./run_iteration.sh 0 4 (runs 5 iterations, producing iter5)

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"

START=${1:?Usage: run_iteration.sh <start_iter> <end_iter> [in_channels]}
END=${2:?Usage: run_iteration.sh <start_iter> <end_iter> [in_channels]}
IN_CHANNELS=${3:-8}

# Configurable parameters for Attempt 8 scaling
NUM_GAMES=${NUM_GAMES:-10000}
N_SIMULATIONS=${N_SIMULATIONS:-128}
TRAIN_STEPS=${TRAIN_STEPS:-2000}
NUM_HIGH_SIMS_GAMES=${NUM_HIGH_SIMS_GAMES:-}
LOW_SIMULATIONS=${LOW_SIMULATIONS:-16}
RESUME=${RESUME:-false}

mkdir -p "${EXP_DIR}/checkpoints"
mkdir -p "${EXP_DIR}/logs"


START_TIME=$(date +%s)

# ------------------ BOOTSTRAP PHASE ------------------
# If we start from iteration 0, we must bootstrap iter0.safetensors by training on random games
if [ "$START" -eq 0 ] && [ ! -f "${EXP_DIR}/checkpoints/iter0.safetensors" ]; then
    echo "=========================================================="
    echo "Bootstrapping Iteration 0: Random Self-play Collection"
    echo "=========================================================="
    t0=$(date +%s)
    BOOTSTRAP_GAMES=$(( NUM_GAMES < 1000 ? NUM_GAMES : 1000 ))
    uv run python "${EXP_DIR}/pre_collect_random.py" \
        --num-games "$BOOTSTRAP_GAMES" \
        --num-workers 8 \
        --seed 42 \
        2>&1 | tee "${EXP_DIR}/logs/bootstrap_collect.log"
    t1=$(date +%s)
    echo "--> Bootstrap game collection took $((t1 - t0)) seconds."

    echo "=========================================================="
    echo "Bootstrapping Iteration 0: Model Training from Scratch"
    echo "=========================================================="
    t0=$(date +%s)
    uv run python "${EXP_DIR}/train.py" \
        --dataset-dir "${EXP_DIR}/random-it0" \
        --resume-from "" \
        --save-checkpoint "${EXP_DIR}/checkpoints/iter0.safetensors" \
        --steps "$TRAIN_STEPS" \
        --lr 1e-3 \
        --batch-size 64 \
        --seed 42 \
        --in-channels "$IN_CHANNELS" \
        2>&1 | tee "${EXP_DIR}/logs/bootstrap_train.log"
    t1=$(date +%s)
    echo "--> Bootstrap training took $((t1 - t0)) seconds."
    
    # Telemetry Sanity Check on Bootstrap Iteration 0
    echo "--> Running Telemetry Health Check on bootstrap iter0..."
    uv run python "${WORKSPACE_ROOT}/scripts/telemetry_alert.py" \
        --checkpoint "${EXP_DIR}/checkpoints/iter0.safetensors" \
        --selfplay-dir "${EXP_DIR}/random-it0" \
        --in-channels "$IN_CHANNELS" \
        --iteration 0 || [ "$NUM_GAMES" -lt 10 ]
    echo ""
fi

# ------------------ AUTO-SURGERY DETECTOR ------------------
# Automatically converts the starting checkpoint from 3-channel to 8-channel if needed
START_CKPT="${EXP_DIR}/checkpoints/iter${START}.safetensors"
if [ -f "$START_CKPT" ]; then
    echo "Checking channels of starting checkpoint ${START_CKPT}..."
    CHANNELS_IN_FILE=$(uv run python -c "import mlx.core as mx; weights = mx.load('${START_CKPT}'); print(weights['input_conv.weight'].shape[3])")
    echo "--> Checkpoint has ${CHANNELS_IN_FILE} channels (target: ${IN_CHANNELS})."
    
    if [ "$CHANNELS_IN_FILE" -ne "$IN_CHANNELS" ]; then
        if [ "$CHANNELS_IN_FILE" -lt "$IN_CHANNELS" ]; then
            echo ""
            echo "=========================================================="
            echo "Auto-Surgery: Converting ${CHANNELS_IN_FILE}-ch checkpoint to ${IN_CHANNELS}-ch..."
            echo "=========================================================="
            SURGERY_SCRIPT="${EXP_DIR}/../../scripts/weight_surgery.py"
            TMP_CKPT="${START_CKPT%.safetensors}_temp_surgery.safetensors"
            
            uv run python "$SURGERY_SCRIPT" --input "$START_CKPT" --output "$TMP_CKPT" --in-channels "$IN_CHANNELS"
            mv "$TMP_CKPT" "$START_CKPT"
            echo "🟢 Auto-Surgery complete! ${START_CKPT} has been expanded to ${IN_CHANNELS}-channel."
            echo ""
        else
            echo "ERROR: Channel mismatch. Checkpoint has ${CHANNELS_IN_FILE} channels, but target is ${IN_CHANNELS}."
            exit 1
        fi
    fi
fi

# ------------------ REINFORCEMENT LEARNING LOOP ------------------
for ITER in $(seq "$START" "$END"); do
    echo "=========================================================="
    echo "Starting Iteration ${ITER}"
    echo "=========================================================="
    
    # 1. Self-Play Collection using iter{ITER}
    CKPT="${EXP_DIR}/checkpoints/iter${ITER}.safetensors"
    DATA_DIR="${EXP_DIR}/selfplay/iter${ITER}"
    
    # Enable Phase 2 opponent pooling conditional
    OPPONENT_POOL_FLAGS=""
    if [ "$ITER" -ge 11 ]; then
        echo "--> Phase 2: Opponent pooling enabled (using checkpoints pool directory)."
        OPPONENT_POOL_FLAGS="--opponent-pool-dir ${EXP_DIR}/checkpoints"
    else
        echo "--> Phase 1: Pure self-play."
    fi

    # Hybrid budget and resumption flags
    HYBRID_FLAGS=""
    if [ -n "${NUM_HIGH_SIMS_GAMES}" ]; then
        echo "--> Hybrid simulation budget: ${NUM_HIGH_SIMS_GAMES} games @ high, remainder @ ${LOW_SIMULATIONS} sims"
        HYBRID_FLAGS="--num-high-sims-games ${NUM_HIGH_SIMS_GAMES} --low-simulations ${LOW_SIMULATIONS}"
    fi

    COLLECT_RESUME_FLAG=""
    if [ "${RESUME}" = "true" ] || [ "${RESUME}" = "1" ]; then
        echo "--> Resumption enabled. collect.py will resume if valid game files exist."
        COLLECT_RESUME_FLAG="--resume"
    fi

    # Configure Playout Cap Randomization (PCR) and Resignation
    PCR_FLAGS=""
    RESIGN_FLAGS=""
    
    if [ "$ITER" -lt 5 ]; then
        echo "--> Warm-up phase (Iter < 5): Resignation and PCR disabled."
    else
        CFG_FILE="${EXP_DIR}/resignation_config.json"
        CURRENT_RESIGN_THRESHOLD="0.02"
        CURRENT_PCR_ENABLED="true"
        
        if [ -f "$CFG_FILE" ]; then
            CURRENT_RESIGN_THRESHOLD=$(uv run python -c "import json; print(json.load(open('${CFG_FILE}'))['resign_threshold'])" 2>/dev/null || echo "0.02")
            CURRENT_PCR_ENABLED=$(uv run python -c "import json; print(str(json.load(open('${CFG_FILE}'))['pcr_enabled']).lower())" 2>/dev/null || echo "true")
        fi
        
        echo "--> Resignation Threshold: ${CURRENT_RESIGN_THRESHOLD}"
        echo "--> PCR Enabled: ${CURRENT_PCR_ENABLED}"
        
        RESIGN_FLAGS="--no-resign-prob 0.10 --resign-threshold ${CURRENT_RESIGN_THRESHOLD}"
        if [ "$CURRENT_PCR_ENABLED" = "true" ]; then
            PCR_FLAGS="--pcr --pcr-low-sims 16 --pcr-high-prob 0.15"
        fi
    fi

    echo "--> Collection: playing ${NUM_GAMES} games with progressive MCTS simulations..."
    t0=$(date +%s)
    uv run python "${EXP_DIR}/collect.py" \
        --checkpoint "$CKPT" \
        --num-games "${NUM_GAMES}" \
        --n-simulations "${N_SIMULATIONS}" \
        --save-dir "$DATA_DIR" \
        --num-workers 8 \
        --seed $((42 + ITER * 100)) \
        --in-channels "$IN_CHANNELS" \
        --progressive-sims \
        ${OPPONENT_POOL_FLAGS} \
        ${HYBRID_FLAGS} \
        ${COLLECT_RESUME_FLAG} \
        ${PCR_FLAGS} \
        ${RESIGN_FLAGS} \
        2>&1 | tee "${EXP_DIR}/logs/collect_iter${ITER}.log"
    t1=$(date +%s)
    echo "--> Game collection took $((t1 - t0)) seconds."
    
    # Run resignation calibration for next iteration if we are at or past iteration 5
    if [ "$ITER" -ge 5 ]; then
        echo "--> Calibrating resignation threshold for the next iteration..."
        uv run python "${EXP_DIR}/calibrate_resignation.py" \
            --save-dir "$DATA_DIR" \
            --target-config "${EXP_DIR}/resignation_config.json" \
            --target-fpr 0.01
    fi
    echo ""
    
    # 2. Train iter{ITER+1} using iter{ITER} as starting weights
    NEXT=$((ITER + 1))
    NEXT_CKPT="${EXP_DIR}/checkpoints/iter${NEXT}.safetensors"
    
    # Replay symlinking: combine current iteration and past two iterations
    REPLAY_DIR="${EXP_DIR}/selfplay/replay_buffer"
    rm -rf "$REPLAY_DIR" && mkdir -p "$REPLAY_DIR"
    echo "--> Symlinking current iteration games to replay buffer..."
    python3 -c "import os, glob; [os.symlink(f, os.path.join('${REPLAY_DIR}', f'iter${ITER}_' + os.path.basename(f))) for f in glob.glob(os.path.join('${DATA_DIR}', 'game_*.npz'))]"
    
    if [ "$ITER" -gt 0 ]; then
        PREV1_DIR="${EXP_DIR}/selfplay/iter$((ITER - 1))"
        if [ -d "$PREV1_DIR" ] && [ -n "$(find "$PREV1_DIR" -name "game_*.npz" -print -quit 2>/dev/null)" ]; then
            echo "--> Symlinking past iteration $((ITER - 1)) games to replay buffer..."
            python3 -c "import os, glob; [os.symlink(f, os.path.join('${REPLAY_DIR}', 'iter$((ITER - 1))_' + os.path.basename(f))) for f in glob.glob(os.path.join('${PREV1_DIR}', 'game_*.npz'))]"
        fi
    fi
    if [ "$ITER" -gt 1 ]; then
        PREV2_DIR="${EXP_DIR}/selfplay/iter$((ITER - 2))"
        if [ -d "$PREV2_DIR" ] && [ -n "$(find "$PREV2_DIR" -name "game_*.npz" -print -quit 2>/dev/null)" ]; then
            echo "--> Symlinking past iteration $((ITER - 2)) games to replay buffer..."
            python3 -c "import os, glob; [os.symlink(f, os.path.join('${REPLAY_DIR}', 'iter$((ITER - 2))_' + os.path.basename(f))) for f in glob.glob(os.path.join('${PREV2_DIR}', 'game_*.npz'))]"
        fi
    fi

    echo "--> Training: optimizing checkpoint iter${NEXT} for ${TRAIN_STEPS} steps..."
    t0=$(date +%s)
    uv run python "${EXP_DIR}/train.py" \
        --dataset-dir "$REPLAY_DIR" \
        --resume-from "$CKPT" \
        --save-checkpoint "$NEXT_CKPT" \
        --steps "$TRAIN_STEPS" \
        --lr 1e-3 \
        --batch-size 64 \
        --seed $((42 + ITER * 200)) \
        --in-channels "$IN_CHANNELS" \
        2>&1 | tee "${EXP_DIR}/logs/train_iter${NEXT}.log"
    t1=$(date +%s)
    echo "--> Training took $((t1 - t0)) seconds."
    
    # Train Sibling Model for diversified league play
    SIB_CKPT="${EXP_DIR}/checkpoints/iter${NEXT}_sibling.safetensors"
    echo "--> Training Sibling Model: optimizing checkpoints iter${NEXT}_sibling..."
    uv run python "${EXP_DIR}/train.py" \
        --dataset-dir "$REPLAY_DIR" \
        --resume-from "$CKPT" \
        --save-checkpoint "$SIB_CKPT" \
        --steps "$TRAIN_STEPS" \
        --lr 2e-3 \
        --batch-size 64 \
        --seed $((142 + ITER * 200)) \
        --in-channels "$IN_CHANNELS" \
        2>&1 | tee "${EXP_DIR}/logs/train_sibling_iter${NEXT}.log"

    
    # 3. Telemetry Sanity Check (Fail-Fast)
    echo "--> Running Telemetry Health Check on iter${NEXT}..."
    uv run python "${WORKSPACE_ROOT}/scripts/telemetry_alert.py" \
        --checkpoint "$NEXT_CKPT" \
        --selfplay-dir "$DATA_DIR" \
        --in-channels "$IN_CHANNELS" \
        --iteration "$NEXT" || [ "$NUM_GAMES" -lt 10 ]
    echo ""

    # 4. Live Evaluation Gate (Fail-Fast)
    # Compare iter${NEXT} against iter${ITER} in a tournament.
    # Win rate must be >= 55.0% for the loop to proceed.
    echo "=========================================================="
    echo "Live Evaluation Gate: Model iter${NEXT} vs Predecessor iter${ITER}"
    echo "=========================================================="
    t0_eval=$(date +%s)
    
    # We will use 40 games (20 Black, 20 White) to evaluate
    EVAL_GATE_GAMES=40
    # For dry-run/mock testing with small NUM_GAMES, we scale it down to 4 games
    if [ "$NUM_GAMES" -lt 100 ]; then
        EVAL_GATE_GAMES=4
    fi
    
    # Run evaluation with D4 ensembling enabled and strict target win rate of >= 55%
    uv run python "${EXP_DIR}/evaluate.py" \
        --checkpoint "$NEXT_CKPT" \
        --opponent-checkpoint "$CKPT" \
        --num-games "$EVAL_GATE_GAMES" \
        --n-simulations 64 \
        --num-workers 8 \
        --seed 1000 \
        --in-channels "$IN_CHANNELS" \
        --d4-ensemble \
        --min-win-rate 55.0 \
        2>&1 | tee "${EXP_DIR}/logs/eval_gate_iter${NEXT}.log"
        
    t1_eval=$(date +%s)
    echo "--> Live Evaluation Gate completed in $((t1_eval - t0_eval)) seconds."
    echo ""
done

# ------------------ EVALUATION PHASE ------------------
FINAL_CKPT="${EXP_DIR}/checkpoints/iter$((END + 1)).safetensors"
OPPONENT_FLAGS=""
OPPONENT_NAME="RandomAgent"

START_CKPT="${EXP_DIR}/checkpoints/iter${START}.safetensors"
if [ -f "$START_CKPT" ]; then
    OPPONENT_FLAGS="--opponent-checkpoint ${START_CKPT}"
    OPPONENT_NAME="iter${START}.safetensors (starting baseline)"
fi

echo "=========================================================="
echo "Final Evaluation: Model ${FINAL_CKPT} vs ${OPPONENT_NAME}"
echo "=========================================================="
t0=$(date +%s)
# Use 10 games for evaluation if NUM_GAMES is small (e.g. in dry-run)
EVAL_GAMES=$(( NUM_GAMES < 100 ? 10 : 100 ))
uv run python "${EXP_DIR}/evaluate.py" \
    --checkpoint "$FINAL_CKPT" \
    ${OPPONENT_FLAGS} \
    --num-games "$EVAL_GAMES" \
    --n-simulations "${N_SIMULATIONS}" \
    --num-workers 8 \
    --seed 1000 \
    --in-channels "$IN_CHANNELS" \
    --d4-ensemble \
    2>&1 | tee "${EXP_DIR}/logs/evaluation.log"
t1=$(date +%s)
echo "--> Evaluation took $((t1 - t0)) seconds."
echo ""

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "=========================================================="
echo "Phase 10 Training & Evaluation Completed Successfully!"
echo "Total execution time: ${TOTAL_ELAPSED} seconds."
echo "=========================================================="
