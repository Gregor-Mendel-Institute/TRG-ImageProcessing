
time python ../processing/processing.py \
  --dpi=13039 \
  --run_ID=inference_run0 \
  --input='../training/sample_dataset/val/2019103107-01(3)_00014016a_m36.tif' \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0.17 \
  --sliding_window_overlap=0.75 \
  --min_mask_overlap=3 \
  --n_detection_rows=1 \
  --cracks=True \
  --debug=True \
  --print_detections=True