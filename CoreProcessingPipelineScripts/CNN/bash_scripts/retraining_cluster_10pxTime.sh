#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=medium
#SBATCH --time=0-10:00:00
#SBATCH --output=retraining_10pxLR0001_E1000_part2.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 \
postprocessingCracksRings.py \
--dataset=/groups/swarts/user/miroslav.polacek/FullSpruceDatasetWithCVAT10pxBuffer  \
--weightRing=/groups/swarts//user/miroslav.polacek/TRG_development0/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/logs/retrainedrings20230821T1558/mask_rcnn_retrainedrings_0125.h5 \

