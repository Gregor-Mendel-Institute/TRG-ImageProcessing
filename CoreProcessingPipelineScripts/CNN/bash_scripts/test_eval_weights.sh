time python ../processing/processing.py \
  --training_data=../training/sample_dataset \
  --weights=../weights/lastYolo12_1024px_1000ep.pt \
  --output_folder=../output \
  --cropUpandDown=0 \
  --sliding_window_overlap=0 \
  --debug \
  --evaluate_weight \
  --run_ID=Test_eval_weights
