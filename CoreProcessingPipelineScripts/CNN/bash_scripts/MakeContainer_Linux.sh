#!/usr/bin/env bash
# Require sudo

mkdir $PWD"/apptainer_cache"
mkdir $PWD"/apptainer_tmp"

export APPTAINER_CACHEDIR=$PWD"/apptainer_cache"
export APPTAINER_TMPDIR=$PWD"/apptainer_tmp"

## in case you need also the def file
source /opt/miniconda3/bin/activate root
conda activate spython
spython recipe $PWD"/../Dockerfile" > $PWD"/../Yolov8.def"

sudo -E apptainer build $PWD"/../image-processing_master.sif" $PWD"/../Yolov8.def"

#rm -r $PWD"/apptainer_cache"
#rm -r $PWD"/apptainer_tmp"