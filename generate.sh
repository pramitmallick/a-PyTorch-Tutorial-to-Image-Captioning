#!/bin/bash
#SBATCH --verbose
#SBATCH --job-name=gencap
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
#SBATCH --gres=gpu:1

python -u generate.py --model='../best_model/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar' --word_map='../best_model/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json' --beam_size=10
#python -u train.py
