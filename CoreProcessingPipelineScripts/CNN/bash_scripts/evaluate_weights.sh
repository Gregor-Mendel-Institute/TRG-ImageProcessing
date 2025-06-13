
time python ../training/evaluate_weights.py \
  --training_data=../training/sample_dataset \
  --output_folder=../output \
  --debug=True \
  --run_ID=Retraining_debug

# --output_folder and --run_ID indicate where to find the weights to evaluate
# this scripts expects directory structure as from our training