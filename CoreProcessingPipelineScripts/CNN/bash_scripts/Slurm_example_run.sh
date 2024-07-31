#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=0-01:00:00
#SBATCH --output=Slurm_example_run.stdout

singularity run --nv image-processing_master.sif \
  --dpi=13039 \
  --run_ID=RUNID \
  --input=../training/sample_dataset/train/1350_00041008a_0_pSX1.965402638101432_pSY1.9654116736824034.tif \
  --output_folder=../output \

###Run like sbatch /groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/processing/measureCoresCracksRingsByPlot.sh 12
