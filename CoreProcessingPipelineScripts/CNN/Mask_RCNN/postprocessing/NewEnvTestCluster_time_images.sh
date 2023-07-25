#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=00-15:00:00
#SBATCH --output=NewEnv_time_test.stdout

ml load build-env/f2022 # required for anaconda3/2023.03
ml load anaconda3/2023.03
source activate ~/.conda/envs/TRGTF2.12P3.11GPUNEW

~/.conda/envs/TRGTF2.12P3.11GPUNEW/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Inf_time_test_tf2_NewEnvCorrectPython \
  --input=/groups/swarts/lab/DendroImages/CNN_test/AlexPOS/MEECNNPaperTreeringSupplementalInfo/Tiffs \
  --weightRing=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=../output \
  --print_detections=no \
