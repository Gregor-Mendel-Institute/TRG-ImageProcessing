#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=64G
#SBATCH --qos=medium
#SBATCH --time=00-20:00:00
#SBATCH --output=TRACE_multispec.stdout

ml load build-env/f2022 # required for anaconda3/2022.05
ml load anaconda3/2022.05
source activate ~/.conda/envs/TRGTF2.12P3.11GPUtesting

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

~/.conda/envs/TRGTF2.12P3.11GPUtesting/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=TRACE_multispec_detect \
  --input=/groups/swarts/lab/TRACE_2023_posterData/TRACE_multispec_dendroelev_adjusted \
  --weightRing=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --output_folder=/groups/swarts/lab/TRACE_2023_posterData/CNN_output \
  --print_detections=yes \
