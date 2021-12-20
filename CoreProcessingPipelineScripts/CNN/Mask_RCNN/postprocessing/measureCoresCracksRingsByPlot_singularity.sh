#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=long
#SBATCH --time=5-00:00:00
#SBATCH --output="/groups/swarts/lab/DendroImages/CNN_results/measureCoresCracksRings$(date).stdout"

STITCH_FOLDER='/groups/swarts/lab/DendroImages/Stitch/'
PLOT=$1
TIFS=$(find ${STITCH_FOLDER} -name "000"$PLOT"*_pS1.945*.tif" )

for i in $TIFS; do
runID_dir=${i%/*}
runID=${runID_dir##*/}
singularity run --nv /scratch-cbe/shared/containers/trg-imageprocessing_master.sif \
  --dpi=13039 \
  --run_ID=$runID \
  --input=$i \
  --weightRing=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --weightCrack=/groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 \
  --output_folder="/groups/swarts/lab/DendroImages/CNN_results/Plot"$PLOT
done


###Run like sbatch /groups/swarts/lab/ImageProcessingPipeline/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/measureCoresCracksRingsByPlot.sh 12
