#!/usr/bin/env bash
# mugo — Phase 8a Iteration Orchestrator (Smoke Test)
# Drives self-play collection and model training for one iteration natively on Mac.
# Usage: ./run_iteration.sh <start_iter> <end_iter>
# Example: ./run_iteration.sh 0 1

set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"

START=${1:?Usage: run_iteration.sh <start_iter> <end_iter>}
END=${2:?Usage: run_iteration.sh <start_iter> <end_iter>}

mkdir -p "${EXP_DIR}/checkpoints"
mkdir -p "${EXP_DIR}/logs"

print_vram() {
    uv run python -c "import mlx.core as mx; print(f'Peak GPU Memory: {mx.metal.get_peak_memory() / (1024**2):.1f} MB')"
}

START_TIME=$(date +%s)

for ITER in $(seq "$START" "$END"); do
    echo "=========================================================="
    echo "Starting Iteration ${ITER}"
    echo "=========================================================="
    
    # ------------------ COLLECTION PHASE ------------------
    if [ "$ITER" -eq 0 ]; then
        CKPT="${EXP_DIR}/checkpoints/iter0.safetensors"
        DATA_DIR="${EXP_DIR}/selfplay/iter0"
        
        echo "--> Phase 8a: Self-Play Game Collection (Iteration 0)"
        echo "Playing 200 games with 64 simulations using MCTS on Apple Silicon..."
        
        t0=$(date +%s)
        uv run python "${EXP_DIR}/collect.py" \
            --checkpoint "$CKPT" \
            --num-games 200 \
            --n-simulations 64 \
            --save-dir "$DATA_DIR" \
            --num-workers 8 \
            --seed 42 \
            2>&1 | tee "${EXP_DIR}/logs/collect_iter${ITER}.log"
        t1=$(date +%s)
        
        echo "--> Game collection took $((t1 - t0)) seconds."
        print_vram
        echo ""
    fi
    
    # ------------------ TRAINING PHASE --------------------
    if [ "$ITER" -eq 0 ]; then
        SRC_CKPT="${EXP_DIR}/checkpoints/iter0.safetensors"
        DST_CKPT="${EXP_DIR}/checkpoints/iter1.safetensors"
        DATA_DIR="${EXP_DIR}/selfplay/iter0"
        
        echo "--> Phase 8a: Model Training (Iteration 1)"
        echo "Training for 300 steps with batch size 64..."
        
        t0=$(date +%s)
        uv run python "${EXP_DIR}/train.py" \
            --dataset-dir "$DATA_DIR" \
            --resume-from "$SRC_CKPT" \
            --save-checkpoint "$DST_CKPT" \
            --lr 1e-3 \
            --batch-size 64 \
            --steps 300 \
            --seed 42 \
            2>&1 | tee "${EXP_DIR}/logs/train_iter1.log"
        t1=$(date +%s)
        
        echo "--> Training took $((t1 - t0)) seconds."
        print_vram
        echo ""
    fi
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "=========================================================="
echo "Smoke Iteration Done!"
echo "Total execution time: ${TOTAL_ELAPSED} seconds."
echo "=========================================================="
