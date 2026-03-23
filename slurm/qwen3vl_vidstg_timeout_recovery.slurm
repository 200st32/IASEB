#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem48
#SBATCH --job-name="qwen3vl-timeout-fix"
#SBATCH --time=0-01:30:00
#SBATCH --output=results/qwen3vl_vidstg_timeout_%A_%a.out
#SBATCH --array=1,2,3,4,9

# Recovery for timed-out arrays from job 323562
# These arrays have partial checkpoints and just need to finish remaining samples
# Array 1: 10 remaining, Array 2: 22 remaining, Array 3: 8 remaining
# Array 4: 4 remaining, Array 9: 5 remaining
# Total: 49 samples, ~1.5h at 2 min/sample

# Original job 323562 parameters
TOTAL_SAMPLES=4149
NUM_JOBS=45
CHUNK_SIZE=93

START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then
    END_IDX=$TOTAL_SAMPLES
fi
ITERS=$(( END_IDX - START_IDX ))

# Use EXISTING checkpoint from job 323562 - will resume from where it left off
CHECKPOINT="results/qwen3vl_vidstg_recovery_323562_${SLURM_ARRAY_TASK_ID}.jsonl"
LOGFILE="results/qwen3vl_vidstg_recovery_323562_${SLURM_ARRAY_TASK_ID}.json"

echo "=== Timeout Recovery for Array $SLURM_ARRAY_TASK_ID ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $LOGFILE"
echo "Sample range: $START_IDX to $END_IDX ($ITERS total)"
echo "Existing checkpoint entries: $(wc -l < $CHECKPOINT)"
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
