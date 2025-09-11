time python ../processing/processing.py \
  --training_data=/Users/miroslav/Documents/Timon_annot/Timon_all_annot_squares/Timon_data_training \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --epochs=1 \
  --generate_annotations=True \
  --annot_buffer=3 \
  --debug=True \
  --run_ID=Debug_training
