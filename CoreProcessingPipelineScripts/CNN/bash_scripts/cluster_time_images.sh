#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --partition=g
#SBATCH --gres=gpu:RTX:1
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=00-20:00:00
#SBATCH --output=Time_images_OldDataset_Yolov8.stdout

ml load build-env/f2022 # required for anaconda3
ml load anaconda3/2023.03
source activate ~/.conda/envs/YOLOv82_P312

~/.conda/envs/YOLOv82_P312/bin/python3 ../processing/processing.py \
  --dpi=13039 \
  --run_ID=Time_images_OldDataset_Yolov8 \
  --input=/groups/swarts/lab/DendroImages/CNN_test/AlexPOS/MEECNNPaperTreeringSupplementalInfo/Tiffs \
  --weightRing=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cracks=False \
  --sliding_window_overlap=0.5 \
  --debug=True \
  --print_detections=yes \
