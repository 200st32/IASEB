#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem48
#SBATCH --job-name="qwen3vl-vidstg-final"
#SBATCH --time=0-03:00:00
#SBATCH --output=results/qwen3vl_vidstg_final_%A_%a.out
#SBATCH --array=14,15,18,23,25,26,27,28,29,33,34,36,39,42

# Final recovery for all 14 incomplete VidSTG arrays
# Total ~203 samples remaining, checkpoint resume will skip already-done ones
# Longest array (34) has 40 remaining => ~2h worst case

TOTAL_SAMPLES=4149
NUM_JOBS=45
CHUNK_SIZE=93

START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then
    END_IDX=$TOTAL_SAMPLES
fi
ITERS=$(( END_IDX - START_IDX ))

CHECKPOINT="results/qwen3vl_vidstg_recovery_323562_${SLURM_ARRAY_TASK_ID}.jsonl"
LOGFILE="results/qwen3vl_vidstg_recovery_323562_${SLURM_ARRAY_TASK_ID}.json"

echo "=== Final Recovery for Array $SLURM_ARRAY_TASK_ID ==="
echo "Checkpoint: $CHECKPOINT ($(wc -l < $CHECKPOINT) entries already done)"
echo "Start: $START_IDX, Iters: $ITERS"
nvidia-smi

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset vidstg \
    --model qwen3vl \
    --task_type freeform \
    --config config.yaml \
    --output_path $LOGFILE \
    --checkpoint_path $CHECKPOINT \
    --device cuda \
    --entry_index $START_IDX \
    --max_iters $ITERS
