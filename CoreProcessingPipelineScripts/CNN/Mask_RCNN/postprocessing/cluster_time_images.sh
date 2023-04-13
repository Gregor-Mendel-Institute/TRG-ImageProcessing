#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=00-20:00:00
#SBATCH --output=Time_images_TF1.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

~/.conda/envs/TreeRingCNN/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Inference_time_test_TF1 \
  --input=/groups/swarts/lab/DendroImages/CNN_test/AlexPOS/MEECNNPaperTreeringSupplementalInfo/Tiffs \
  --weightRing=/users/miroslav.polacek/TRG_testing/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/onlyring/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=../output \
