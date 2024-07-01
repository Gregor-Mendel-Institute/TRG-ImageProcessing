#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=00-04:00:00
#SBATCH --output=Detect_new_training_squares.stdout

ml load build-env/f2022 # required for anaconda3
ml load anaconda3/2023.03
source activate ~/.conda/envs/YOLOv82_P312

~/.conda/envs/YOLOv82_P312/bin/python3 ../postprocessing/postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=check_training_squares \
  --input=/groups/swarts/user/miroslav.polacek/CNN_yolov8_retraining_data/chopped \
  --weightRing=../weights/best10pxlowerlrf.pt \
  --output_folder=../output \
  --cracks=True \
  --print_detections=yes \
