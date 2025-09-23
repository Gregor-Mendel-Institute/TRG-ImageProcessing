#!/usr/bin/env bash

mkdir $PWD"/apptainer_cache"
mkdir $PWD"/apptainer_tmp"

export APPTAINER_CACHEDIR=$PWD"/apptainer_cache"
export APPTAINER_TMPDIR=$PWD"/apptainer_tmp"

spython recipe Dockerfile > Yolov8.def
apptainer build --fakeroot image-processing_master.sif Yolov8.def

rm -r ./apptainer_cache
rm -r ./apptainer_tmp