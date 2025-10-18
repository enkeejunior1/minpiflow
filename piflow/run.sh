#!/bin/bash
#SBATCH --job-name=piflow
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00

# Environment setup
module purge
source /vast/projects/jgu32/lab/yhpark/miniconda3/etc/profile.d/conda.sh
conda activate cvpr
cd $SLURM_SUBMIT_DIR

# Actual work
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# python3 piflow.py --dataset cifar --NFE 4 --K 8
# python3 piflow.py --dataset mnist --NFE 4 --K 8 --iter 8 --debug
python3 piflow.py --dataset mnist --NFE 1 --K 8 --iter 8 --debug