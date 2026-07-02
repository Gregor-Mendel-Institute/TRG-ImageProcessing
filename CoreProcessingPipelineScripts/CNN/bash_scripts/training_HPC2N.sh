#!/bin/bash
#SBATCH -A hpc2n2024-106
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH -C nvidia_gpu
#SBATCH --output=Training.%J.out

module purge  > /dev/null 2>&1
ml GCC/13.3.0 OpenMPI/5.0.3 OpenCV/4.11.0-CUDA-12.6.0-contrib
source ~/venvs/ultra831P3123/bin/activate

time python ../processing/processing.py \
  --training_data=/proj/nobackup/trgstor/user/miroslav.polacek/UpdatedDatasetOld_Third_Timon \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --epochs=1 \
  --annot_buffer=10 \
  --debug \
  --run_ID=Training_fullwithTimonyolov8

# --generate_annotations \