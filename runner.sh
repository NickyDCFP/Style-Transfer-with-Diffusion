#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=0.2_AFHQ_Diffusion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pa2439@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:4

module purge
module load anaconda3/2020.07;
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate ./penv
export PATH=./penv/bin:$PATH;

torchrun --nproc_per_node=4 main.py --dataset afhq --arch UNet --class-cond --epochs 500 --batch-size 128 --sampling-steps 50 --data-dir "./data/afhq/train" --lambda_cyc=0.2
