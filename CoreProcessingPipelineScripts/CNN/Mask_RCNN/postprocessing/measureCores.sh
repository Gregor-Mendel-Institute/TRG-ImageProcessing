#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --mem=64G
#SBATCH --qos=medium
#SBATCH --time=1-00:00:00
#SBATCH --output=measureCoresTest.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN



~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/lab/ImageProcessingPipeline/CNN/Mask_RCNN/postprocessing/postprocessing.py --dpi=13039 --image=/groups/swarts/lab/DendroImages/Plot8CNN_toTest/ --weight=/groups/swarts/lab/ImageProcessingPipeline/CNN/Mask_RCNN/logs/traintestmlw220200601T2039/mask_rcnn_traintestmlw2_0293.h5 --output_folder=/groups/swarts/lab/ImageProcessingPipeline/CNN/Mask_RCNN/TestOutput/

