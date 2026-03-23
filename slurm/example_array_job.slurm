#!/bin/bash
#SBATCH -J iaseb_array
#SBATCH -p gpu                       # <<< CHANGE: Your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -C gmem48                    # <<< CHANGE: Your GPU memory constraint
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-9                  # <<< CHANGE: Number of array tasks
#SBATCH -o results/%x_%A_%a.out

# <<< CHANGE: Update module loads and conda env for your cluster
module load cuda/12.1
source activate iaseb

DATASET="hcstvg1"                    # <<< CHANGE: dataset name
MODEL="qwen3vl"                      # <<< CHANGE: model name
TASK_TYPE="freeform"                 # <<< CHANGE: freeform or referral
NUM_JOBS=10                          # <<< CHANGE: must match --array count

# Calculate the slice for this array task
TOTAL_SAMPLES=$(python -c "
from IASEB.datasets import STVGDataLoader, load_dataset_config
import argparse
load_dataset_config('config.yaml')
args = argparse.Namespace(dataset='$DATASET', task_type='$TASK_TYPE', frame_step=5)
print(len(STVGDataLoader(args)))
")

SAMPLES_PER_JOB=$(( (TOTAL_SAMPLES + NUM_JOBS - 1) / NUM_JOBS ))
START_INDEX=$(( SLURM_ARRAY_TASK_ID * SAMPLES_PER_JOB ))

OUTPUT_PATH="results/${MODEL}_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
CHECKPOINT_PATH="results/${MODEL}_${DATASET}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.jsonl"

python -m IASEB.run_eval \
    --dataset $DATASET \
    --model $MODEL \
    --task_type $TASK_TYPE \
    --config config.yaml \
    --output_path $OUTPUT_PATH \
    --checkpoint_path $CHECKPOINT_PATH \
    --device cuda \
    --frame_step 5 \
    --entry_index $START_INDEX \
    --max_iters $SAMPLES_PER_JOB
