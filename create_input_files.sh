#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=captioning
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python create_input_files.py
