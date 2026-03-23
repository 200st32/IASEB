#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem48
#SBATCH --job-name="qwen3vl-timeout-fix2"
#SBATCH --time=0-01:00:00
#SBATCH --output=results/qwen3vl_vidstg_timeout2_%A_%a.out
#SBATCH --array=14,15

# Recovery for arrays 14,15 that also timed out
# Array 14: 16 remaining, Array 15: 17 remaining (~33 samples, ~1h)

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

echo "=== Timeout Recovery 2 for Array $SLURM_ARRAY_TASK_ID ==="
echo "Checkpoint: $CHECKPOINT ($(wc -l < $CHECKPOINT) entries)"
nvidia-smi

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset vidstg \
    --model qwen3vl \
    --task_type freeform \
    --output_path $LOGFILE \
    --checkpoint_path $CHECKPOINT \
    --device cuda \
    --entry_index $START_IDX \
    --max_iters $ITERS
