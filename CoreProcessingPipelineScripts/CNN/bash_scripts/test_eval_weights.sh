time python ../training/evaluate_weight.py \
  --training_data=../training/sample_dataset \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0 \
  --sliding_window_overlap=0.75 \
  --debug=True \
  --run_ID=Test_eval_weights
