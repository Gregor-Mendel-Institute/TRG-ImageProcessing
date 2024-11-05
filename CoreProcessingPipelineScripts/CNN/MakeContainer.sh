#!/usr/bin/env bash

export APPTAINER_CACHEDIR=/home/mirosonia/Documents/apptainer_cache
export APPTAINER_TMPDIR=/home/mirosonia/Documents/apptainer_tmp

spython recipe Dockerfile > Yolov8.def
apptainer build image-processing_master.sif Yolov8.def
