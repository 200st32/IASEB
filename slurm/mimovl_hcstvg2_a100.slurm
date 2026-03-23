#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem80
#SBATCH --job-name="mimo-hc2"
#SBATCH --time=0-06:00:00
#SBATCH --output=results/mimovl_hcstvg2_%A_%a.out
#SBATCH --array=0-4

DATASET="hcstvg2"
MODEL="mimovl"
TOTAL_SAMPLES=2000
NUM_JOBS=5
CHUNK_SIZE=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))

START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END_IDX=$(( START_IDX + CHUNK_SIZE ))
if [ $END_IDX -gt $TOTAL_SAMPLES ]; then END_IDX=$TOTAL_SAMPLES; fi
ITERS=$(( END_IDX - START_IDX ))

OUTPUT="results/mimovl_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
CHECKPOINT="results/mimovl_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

echo "=== MiMo-VL $DATASET Array $SLURM_ARRAY_TASK_ID ==="
echo "Samples: $START_IDX to $((START_IDX + ITERS)) ($ITERS total)"
echo "Start: $(date)"
nvidia-smi

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset $DATASET \
    --model $MODEL \
    --task_type freeform \
    --config config.yaml \
    --output_path $OUTPUT \
    --checkpoint_path $CHECKPOINT \
    --device cuda \
    --entry_index $START_IDX \
    --max_iters $ITERS

echo "End: $(date)"
