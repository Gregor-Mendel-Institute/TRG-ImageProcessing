#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=0-08:00:00
#SBATCH --output=measureCores.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 \
  /groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=20210310StitchTest \
  --input=/groups/swarts/lab/DendroImages/Tiles/OldAndTesting/20210310StitchTest \
  --weightRing=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --weightCrack=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 \
  --output_folder=/groups/swarts/lab/DendroImages/CNN_test
