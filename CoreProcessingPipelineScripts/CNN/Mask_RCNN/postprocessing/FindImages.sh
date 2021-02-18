#!/bin/bash

STITCH_FOLDER='/home/miroslavp/Documents/BashScriptTest'
PLOT=14
echo $STITCH_FOLDER
echo $PLOT

TIFS=$(find ${STITCH_FOLDER} -name "000$PLOT*_pSX*TEST*.tif")
echo $TIFS
source activate ~/anaconda3/envs/TreeRingCNN

cd /home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing
python3 postprocessingCracksRings.py --dpi=12926 --run_ID=RUN_ID_SOME_VALUE --input=$TIFS --weightRing=/home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 --weightCrack=/home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 --output_folder=/home/miroslavp/Documents/CNNTestRuns
