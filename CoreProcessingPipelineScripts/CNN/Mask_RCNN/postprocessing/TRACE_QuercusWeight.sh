#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=200G
#SBATCH --qos=medium
#SBATCH --time=00-10:00:00
#SBATCH --output=TRACE_QuercusWeight.stdout

ml load build-env/f2022 # required for anaconda3/2022.05
ml load anaconda3/2022.05
source activate ~/.conda/envs/TRGTF2.12P3.11GPUtesting

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

~/.conda/envs/TRGTF2.12P3.11GPUtesting/bin/python3 postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=TRACE_QuercusWeight \
  --input=/groups/swarts/lab/TRACE_2023_posterData/TRACE_multispec_dendroelev_adjusted/Quercus_only \
  --weightRing=/groups/swarts/user/miroslav.polacek/TRG_development0/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/logs/retrainedrings20230427T2201/mask_rcnn_retrainedrings_0184.h5 \
  --output_folder=/groups/swarts/lab/TRACE_2023_posterData/CNN_output \
  --print_detections=yes \

# correct --dpi=19812 for gigapixel fotos from dendro elevator