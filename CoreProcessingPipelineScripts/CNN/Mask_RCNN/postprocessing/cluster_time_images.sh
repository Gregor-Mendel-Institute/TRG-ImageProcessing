#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=00-02:00:00
#SBATCH --output=Time_images.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

~/.conda/envs/TreeRingCNN/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Image_time_test \
  --input=/groups/swarts/lab/DendroImages/CNN_test/AlexPOS/MEECNNPaperTreeringSupplementalInfo/Tiff \
  --weightRing=/home/miroslavp/Github/test/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/onlyring/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=/home/miroslavp/Pictures/new_val_detections/curent_debug \
  --n_detection_rows=1 \
  --cropUpandDown=0.17 \
  --sliding_window_overlap=0.75 \
  --min_mask_overlap=3 \
  --print_detections=yes \
