#!/usr/bin/env bash

#SBATCH --partition=g
#SBATCH --gpus-per-task=RTX:1
#SBATCH --mem=64G
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --output=measureCoresCracksRingsParallel.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

STITCH_FOLDER='/groups/swarts/lab/DendroImages/Stitch/'
PLOT=14
TIFS=$(find ${STITCH_FOLDER} -name "000$PLOT*_pSX*TEST*.tif")

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/lab/ImageProcessingPipeline/TreeCNN/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/postprocessingCracksRings.py --dpi=13039 --run_ID= --input== --weightRing= --weightCrack= --output_folder=
