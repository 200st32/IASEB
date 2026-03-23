#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem80
#SBATCH --job-name="qwen3vl-vidstg-rec"
#SBATCH --time=0-04:00:00
#SBATCH --output=results/qwen3vl_vidstg_recovery_%A_%a.out
#SBATCH --array=0-44
#SBATCH --exclude=c1-7

# Recovery job for VidSTG arrays 0-8 (samples 0-4149)
# Original arrays 0-8 each had 461 samples, total 4149 samples
# Split into 45 sub-arrays of ~92 samples each (fits in 4h)
# Array 9 (samples 4149-4609) is still running separately

MODEL="qwen3vl"
DATASET="vidstg"
TASK="freeform"
TOTAL_SAMPLES=4149  # Only arrays 0-8 (461 * 9 = 4149)
NUM_JOBS=45

# Calculate chunk size and indices
CHUNK_SIZE=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then
    END_IDX=$TOTAL_SAMPLES
fi
ITERS=$(( END_IDX - START_IDX ))

LOGFILE="results/${MODEL}_${DATASET}_recovery_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
CHECKPOINT="results/${MODEL}_${DATASET}_recovery_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "Job $SLURM_ARRAY_TASK_ID: Processing samples $START_IDX to $END_IDX ($ITERS samples)"
echo "Output: $LOGFILE"
echo "Checkpoint: $CHECKPOINT"
nvidia-smi

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset $DATASET \
    --model $MODEL \
    --task_type $TASK \
    --output_path $LOGFILE \
    --checkpoint_path $CHECKPOINT \
    --device cuda \
    --entry_index $START_IDX \
    --max_iters $ITERS

echo "Job $SLURM_ARRAY_TASK_ID complete. Check $LOGFILE for results."
