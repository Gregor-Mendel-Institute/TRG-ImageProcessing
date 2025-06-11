#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=medium
#SBATCH --time=0-20:00:00
#SBATCH --output=evaluate_training_10px.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 \
evaluate_weights.py \
--dataset=/groups/swarts/user/miroslav.polacek/FullSpruceDatasetWithCVAT10pxBuffer  \
--train_log=../postprocessing/logs/retrainedrings20230822T1435 \

