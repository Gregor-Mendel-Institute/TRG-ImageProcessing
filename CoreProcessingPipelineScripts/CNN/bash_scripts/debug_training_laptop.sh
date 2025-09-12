time python ../processing/processing.py \
  --training_data=../training/sample_dataset \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --epochs=1 \
  --generate_annotations=False \
  --annot_buffer=3 \
  --debug=True \
  --run_ID=Debug_training

#--training_data=../training/sample_dataset \