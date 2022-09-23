#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=00-20:00:00
#SBATCH --output=Time_images.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

~/.conda/envs/TreeRingCNN/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Image_time_test_debug \
  --input=/groups/swarts/lab/DendroImages/CNN_test/AlexPOS/MEECNNPaperTreeringSupplementalInfo/Tiffs \
  --weightRing=/users/miroslav.polacek/TRG_testing/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/onlyring/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=/users/miroslav.polacek/Time_results \
