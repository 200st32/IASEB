#!/bin/bash
#SBATCH -J iaseb_eval
#SBATCH -p gpu                       # <<< CHANGE: Your GPU partition
#SBATCH --gres=gpu:1
#SBATCH -C gmem48                    # <<< CHANGE: Your GPU memory constraint (gmem48 or gmem80)
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o results/%x_%j.out

# <<< CHANGE: Update module loads and conda env for your cluster
module load cuda/12.1
source activate iaseb

DATASET="hcstvg1"                    # <<< CHANGE: hcstvg1, hcstvg2, vidstg, vidvrd, mevis, rvos
MODEL="qwen3vl"                      # <<< CHANGE: cogvlm, shikra, ferret, qwen3vl
TASK_TYPE="freeform"                 # <<< CHANGE: freeform or referral
OUTPUT_PATH="results/${MODEL}_${DATASET}_${SLURM_JOB_ID}.json"

python -m IASEB.run_eval \
    --dataset $DATASET \
    --model $MODEL \
    --task_type $TASK_TYPE \
    --config config.yaml \
    --output_path $OUTPUT_PATH \
    --device cuda \
    --frame_step 5
