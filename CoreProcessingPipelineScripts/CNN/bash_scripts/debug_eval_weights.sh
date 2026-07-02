time python ../processing/processing.py \
  --training_data=../training/sample_dataset \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0 \
  --sliding_window_overlap=0.75 \
  --debug \
  --evaluate_weight \
  --run_ID=debug_new_get_metrics_all_sample_im_0
