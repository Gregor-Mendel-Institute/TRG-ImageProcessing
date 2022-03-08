#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=long
#SBATCH --time=8-00:00:00
#SBATCH --output=Train_onlyRings.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 \
Train_Rings.py \
train --dataset=sample_dataset  \
--weights=imagenet
