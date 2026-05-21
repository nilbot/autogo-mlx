#!/usr/bin/env bash
# mugo — Phase 10 Iteration Orchestrator
# Usage: ./run_iteration.sh <start_iter> <end_iter>
# Example: ./run_iteration.sh 0 4 (runs 5 iterations, producing iter5)

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

# ------------------ BOOTSTRAP PHASE ------------------
# If we start from iteration 0, we must bootstrap iter0.safetensors by training on random games
if [ "$START" -eq 0 ] && [ ! -f "${EXP_DIR}/checkpoints/iter0.safetensors" ]; then
    echo "=========================================================="
    echo "Bootstrapping Iteration 0: Random Self-play Collection"
    echo "=========================================================="
    t0=$(date +%s)
    uv run python "${EXP_DIR}/pre_collect_random.py" \
        --num-games 1000 \
        --num-workers 8 \
        --seed 42 \
        2>&1 | tee "${EXP_DIR}/logs/bootstrap_collect.log"
    t1=$(date +%s)
    echo "--> Bootstrap game collection took $((t1 - t0)) seconds."
    print_vram
    echo ""

    echo "=========================================================="
    echo "Bootstrapping Iteration 0: Model Training from Scratch"
    echo "=========================================================="
    t0=$(date +%s)
    uv run python "${EXP_DIR}/train.py" \
        --dataset-dir "${EXP_DIR}/random-it0" \
        --resume-from "" \
        --save-checkpoint "${EXP_DIR}/checkpoints/iter0.safetensors" \
        --steps 2000 \
        --lr 1e-3 \
        --batch-size 64 \
        --seed 42 \
        2>&1 | tee "${EXP_DIR}/logs/bootstrap_train.log"
    t1=$(date +%s)
    echo "--> Bootstrap training took $((t1 - t0)) seconds."
    print_vram
    echo ""
fi

# ------------------ REINFORCEMENT LEARNING LOOP ------------------
for ITER in $(seq "$START" "$END"); do
    echo "=========================================================="
    echo "Starting Iteration ${ITER}"
    echo "=========================================================="
    
    # 1. Self-Play Collection using iter{ITER}
    CKPT="${EXP_DIR}/checkpoints/iter${ITER}.safetensors"
    DATA_DIR="${EXP_DIR}/selfplay/iter${ITER}"
    
    echo "--> Collection: playing 1000 games with 64 MCTS simulations..."
    t0=$(date +%s)
    uv run python "${EXP_DIR}/collect.py" \
        --checkpoint "$CKPT" \
        --num-games 1000 \
        --n-simulations 64 \
        --save-dir "$DATA_DIR" \
        --num-workers 8 \
        --seed $((42 + ITER * 100)) \
        2>&1 | tee "${EXP_DIR}/logs/collect_iter${ITER}.log"
    t1=$(date +%s)
    echo "--> Game collection took $((t1 - t0)) seconds."
    print_vram
    echo ""
    
    # 2. Train iter{ITER+1} using iter{ITER} as starting weights
    NEXT=$((ITER + 1))
    NEXT_CKPT="${EXP_DIR}/checkpoints/iter${NEXT}.safetensors"
    
    echo "--> Training: optimizing checkpoint iter${NEXT} for 2000 steps..."
    t0=$(date +%s)
    uv run python "${EXP_DIR}/train.py" \
        --dataset-dir "$DATA_DIR" \
        --resume-from "$CKPT" \
        --save-checkpoint "$NEXT_CKPT" \
        --steps 2000 \
        --lr 1e-3 \
        --batch-size 64 \
        --seed $((42 + ITER * 200)) \
        2>&1 | tee "${EXP_DIR}/logs/train_iter${NEXT}.log"
    t1=$(date +%s)
    echo "--> Training took $((t1 - t0)) seconds."
    print_vram
    echo ""
done

# ------------------ EVALUATION PHASE ------------------
FINAL_CKPT="${EXP_DIR}/checkpoints/iter$((END + 1)).safetensors"
echo "=========================================================="
echo "Final Evaluation: Model ${FINAL_CKPT} vs RandomAgent"
echo "=========================================================="
t0=$(date +%s)
uv run python "${EXP_DIR}/evaluate.py" \
    --checkpoint "$FINAL_CKPT" \
    --num-games 100 \
    --n-simulations 64 \
    --num-workers 8 \
    --seed 1000 \
    2>&1 | tee "${EXP_DIR}/logs/evaluation.log"
t1=$(date +%s)
echo "--> Evaluation took $((t1 - t0)) seconds."
print_vram
echo ""

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

echo "=========================================================="
echo "Phase 10 Training & Evaluation Completed Successfully!"
echo "Total execution time: ${TOTAL_ELAPSED} seconds."
echo "=========================================================="
