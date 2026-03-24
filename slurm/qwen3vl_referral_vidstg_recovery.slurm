#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem80
#SBATCH --job-name="q3vl-r-stg"
#SBATCH --time=0-08:00:00
#SBATCH --output=results/qwen3vl_referral_vidstg_%A_%a.out
#SBATCH --array=0-4

DATASET="vidstg"
MODEL="qwen3vl"
TASK_TYPE="referral"
TOTAL_SAMPLES=1227
NUM_JOBS=5
CHUNK_SIZE=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))

START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then END_IDX=$TOTAL_SAMPLES; fi
ITERS=$(( END_IDX - START_IDX ))

OUTPUT="results/qwen3vl_${TASK_TYPE}_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
CHECKPOINT="results/qwen3vl_${TASK_TYPE}_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

# Copy merged checkpoint so this array resumes from all prior work
cp results/qwen3vl_referral_vidstg_merged.jsonl "$CHECKPOINT"

echo "=== Qwen3-VL $TASK_TYPE $DATASET Recovery Array $SLURM_ARRAY_TASK_ID ==="
echo "Samples: $START_IDX to $((START_IDX + ITERS)) ($ITERS total)"
nvidia-smi

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

IASEB_QWEN3VL_PATH=/home/aparcedo/Qwen3-VL-8B-Instruct \
/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset $DATASET \
    --model $MODEL \
    --task_type $TASK_TYPE \
    --config config.yaml \
    --output_path $OUTPUT \
    --checkpoint_path $CHECKPOINT \
    --device cuda \
    --entry_index $START_IDX \
    --max_iters $ITERS
