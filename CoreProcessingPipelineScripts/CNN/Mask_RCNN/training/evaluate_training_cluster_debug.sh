#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=short
#SBATCH --time=0-04:00:00
#SBATCH --output=evaluate_training_debug.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 \
evaluate_weights.py \
--dataset=/groups/swarts/user/miroslav.polacek/CNN/TiffsCorrected30  \
--train_log=../postprocessing/logs/retrainedrings20230428T1728 \

