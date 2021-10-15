#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --time=8-00:00:00
#SBATCH --output=measureCoresCracksRings_old_weights.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

STITCH_FOLDER='/groups/swarts/lab/DendroImages/Stitch/'
PLOT=14
TIFS=$(find ${STITCH_FOLDER} -name "000$PLOT*_pSX*.tif" -o -name "000$PLOT*_pS1.9*.tif" )

for i in $TIFS; do
runID_dir=${i%/*}
runID=${runID_dir##*/}
~/.conda/envs/TreeRingCNN/bin/python3 \
  /groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=$runID \
  --input=$i \
  --weightRing=/groups/swarts/lab/ImageProcessingPipeline/TreeCNN/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs//traintestmlw220200727T1332/mask_rcnn_traintestmlw2_0957.h5 \
  --weightCrack=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 \
  --output_folder=/groups/swarts/lab/DendroImages/CNN_test_old_weights
done
