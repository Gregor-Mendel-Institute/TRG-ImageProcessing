#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=64G
#SBATCH --qos=medium
#SBATCH --time=00-20:00:00
#SBATCH --output=TRACE_QuercusWeights.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

~/.conda/envs/TreeRingCNN/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=TRACE_QuercusWeight_detect \
  --input=/groups/swarts/lab/TRACE_2023_posterData/TRACE_multispec_dendroelev_adjusted \
  --weightRing=/groups/swarts/user/miroslav.polacek/TRG_development0/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/logs/retrainedrings20230427T2201/mask_rcnn_retrainedrings_0184.h5 \
  --output_folder=/groups/swarts/lab/TRACE_2023_posterData/CNN_output \
  --print_detections=yes \
