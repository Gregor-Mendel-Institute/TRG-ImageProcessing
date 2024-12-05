#!/bin/bash
#SBATCH -A hpc2n2024-142 # Change to your own
#SBATCH --time=00:30:00  # Asking for 30 minutes
# Asking for one V100 card
#SBATCH --gpus=1
#SBATCH -C nvidia_gpu
#SBATCH --output=Inference_debug.%J.out

module purge  > /dev/null 2>&1
ml GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 scikit-image/0.22.0
source ~/venvs/yoloP311/bin/activate

time python ../processing/processing.py \
  --dpi=13039 \
  --run_ID=Debug \
  --input=../training/sample_dataset/val \
  --weightRing=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0 \
  --sliding_window_overlap=0 \
  --cracks=True \
  --debug=True \
  --print_detections=True