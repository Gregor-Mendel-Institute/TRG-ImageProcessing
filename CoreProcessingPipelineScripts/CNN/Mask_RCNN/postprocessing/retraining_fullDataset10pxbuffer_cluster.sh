#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=long
#SBATCH --time=4-00:00:00
#SBATCH --output=retraining_10pxBuffer.stdout

ml load build-env/f2022 # required for anaconda3/2022.05
ml load anaconda3/2023.03
source activate ~/.conda/envs/TRGTF2.12P3.11GPUNEW

time ~/.conda/envs/TRGTF2.12P3.11GPUNEW/bin/python3 \
postprocessingCracksRings.py \
--dataset=/groups/swarts/user/miroslav.polacek/FullSpruceDatasetWithCVAT10pxBuffer \
--weightRing=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
--start_new=True \
