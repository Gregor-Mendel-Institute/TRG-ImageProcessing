#!/usr/bin/env bash

#SBATCH --partition=g
#SBATCH --gres=gpu:5
#SBATCH --mem=160G
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --output=measureCoresCracksRingsParallel.stdout

ml load anaconda3/2019.03
ml load build-env/2020
ml load parallel/20190222-gcccore-7.3.0
source activate ~/.conda/envs/TreeRingCNN

STITCH_FOLDER='/groups/swarts/lab/DendroImages/Stitch/'
PLOT=14
TIFS=$(find ${STITCH_FOLDER} -name "000$PLOT*_pSX*.tif")

parallel --delay 0.2 --jobs 5 srun -n1 --exclusive ~/.conda/envs/TreeRingCNN/bin/python3 \
  /groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=NO_RUN_ID \
  --input=${TIFS} \
  --weightRing=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --weightCrack=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 \
  --output_folder=/groups/swarts/lab/DendroImages/Plot14_CNN_test
