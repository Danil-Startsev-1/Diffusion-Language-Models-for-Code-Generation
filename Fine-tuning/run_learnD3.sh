#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=2      
#SBATCH --time=6-00:00:00
#SBATCH --job-name=learnD3
#SBATCH --output=/home/dstartsev/checkpoint/D3/learnD3_%j.log

cd /home/dstartsev/checkpoint
source ~/.bashrc
conda activate hf-download

python learnD3.py 