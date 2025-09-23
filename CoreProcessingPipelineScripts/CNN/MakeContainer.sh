#!/usr/bin/env bash

mkdir ./apptainer_cache
mkdir ./apptainer_tmp

export APPTAINER_CACHEDIR=./apptainer_cache
export APPTAINER_TMPDIR=./apptainer_tmp

spython recipe Dockerfile > Yolov8.def
apptainer build image-processing_master.sif Yolov8.def

rm -r ./apptainer_cache
rm -r ./apptainer_tmp