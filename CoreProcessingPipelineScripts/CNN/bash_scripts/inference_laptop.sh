
time python ../processing/processing.py \
  --dpi=13039 \
  --run_ID=Debugging_centerlines \
  --input=../training/sample_dataset/val/ \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0.17 \
  --sliding_window_overlap=0.75 \
  --min_mask_overlap=3 \
  --n_detection_rows=1 \
  --cracks=True \
  --debug=True \
  --print_detections=True

#--input=/Volumes/Storage/Ring_test_examples/mini_ring_examples/to_use \
#--input=/Users/miroslav.polacek/Downloads/TC_selection_of_cores/F4b_B1.png \
#--input=/Users/miroslav.polacek/Pictures/TC_selection_of_cores/cropped/ \
#--input=/Volumes/T7\ Shield/CNN_yolov8_retraining_data/Plot23_badsamplequality/chopped/00023007a_mo1973_pS1.96536799834280303030_3784.tif \
#--input=/Users/miroslav.polacek/Pictures/23_weird_image \
#--input=/Volumes/Storage/Ring_test_examples/whole_core_examples/the_real_ones/00019002a_mo4724_pS1.96536799834280303030.tif \
#--cropUpandDown=0 \
#--sliding_window_overlap=0.5 \