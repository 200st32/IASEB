#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH -C gmem48
#SBATCH --job-name="mimo-timing"
#SBATCH --time=0-00:30:00
#SBATCH --output=results/mimovl_timing_%j.out

# Timing test: 5 samples on hcstvg1 to estimate per-sample cost

module load anaconda3/2019.03-1
module load cuda/12.1
source activate stvg

echo "=== MiMo-VL Timing Test ==="
echo "Start: $(date)"
nvidia-smi

/home/aparcedo/.conda/envs/stvg/bin/python -m IASEB.run_eval \
    --dataset hcstvg1 \
    --model mimovl \
    --task_type freeform \
    --config config.yaml \
    --output_path results/mimovl_timing_test.json \
    --device cuda \
    --entry_index 0 \
    --max_iters 5

echo "End: $(date)"
