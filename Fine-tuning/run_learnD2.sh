#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=2      
#SBATCH --time=6-00:00:00
#SBATCH --job-name=learnD2
#SBATCH --output=/home/dstartsev/checkpoint/D2/learnD2_%j.log

cd /home/dstartsev/checkpoint
source ~/.bashrc
conda activate hf-download

python learnD2.py