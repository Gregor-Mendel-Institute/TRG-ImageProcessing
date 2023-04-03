#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=00-04:00:00
#SBATCH --output=Inference_debug.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

~/.conda/envs/TreeRingCNN/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Inference_debug \
  --input=/groups/swarts/user/miroslav.polacek/TRG-ImageProcessingTESTING/CoreProcessingPipelineScripts/CNN/Mask_RCNN/Data/General_testing_images \
  --weightRing=/users/miroslav.polacek/TRG_testing/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/onlyring/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=../output \
  --print_detections=yes \
