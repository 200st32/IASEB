#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem48
#SBATCH --job-name="mimovl-hcstvg2"
#SBATCH --time=0-06:00:00
#SBATCH --output=results/mimovl_hcstvg2_%A_%a.out
#SBATCH --array=0-9
#SBATCH --exclude=c1-7

MODEL="mimovl"
DATASET="hcstvg2"
TASK="freeform"
TOTAL_SAMPLES=2000
NUM_JOBS=10

# Calculate chunk size and indices
CHUNK_SIZE=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then
    END_IDX=$TOTAL_SAMPLES
fi
ITERS=$(( END_IDX - START_IDX ))

LOGFILE="results/${MODEL}_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "Job $SLURM_ARRAY_TASK_ID: Processing samples $START_IDX to $END_IDX ($ITERS samples)"
echo "Output: $LOGFILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset $DATASET \
    --model $MODEL \
    --task_type $TASK \
    --output_path $LOGFILE \
    --device cuda \
    --entry_index $START_IDX \
    --max_iters $ITERS

echo "Job $SLURM_ARRAY_TASK_ID complete. Check $LOGFILE for results."
