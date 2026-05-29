#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=2      
#SBATCH --time=1-00:00:00
#SBATCH --job-name=testD4
#SBATCH --output=/home/dstartsev/checkpoint/D2/test/testD4_%j.log

cd /home/dstartsev/checkpoint/D2/test
source ~/.bashrc
conda activate hf-download

python testD2.py 