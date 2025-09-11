time python ../processing/processing.py \
  --training_data=../training/sample_dataset \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --epochs=1 \
  --generate_annotations=True \
  --annot_buffer=10 \
  --debug=True \
  --run_ID=Test_training
