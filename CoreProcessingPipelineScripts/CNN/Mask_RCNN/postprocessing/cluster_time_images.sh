#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=00-20:00:00
#SBATCH --output=Inference_time_test.stdout

ml load build-env/f2022 # required for anaconda3/2022.05
ml load anaconda3/2022.05
source activate ~/.conda/envs/TRGTF2.12P3.11GPUtesting

~/.conda/envs/TRGTF2.12P3.11GPUtesting/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Inf_time_test_tf2 \
  --input=/groups/swarts/lab/DendroImages/CNN_test/AlexPOS/MEECNNPaperTreeringSupplementalInfo/Tiffs \
  --weightRing=/users/miroslav.polacek/TRG_testing/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/onlyring/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=../output \
