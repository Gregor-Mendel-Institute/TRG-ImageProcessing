
time python ../processing/processing.py \
  --dpi=13039 \
  --run_ID=debug_detection_formating  \
  --input="../training/sample_dataset/train/14610_00014006b_0_pSX1.9653764466903185_pSY1.9665786978105748.tif" \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0 \
  --sliding_window_overlap=0.75 \
  --cracks \
  --debug \
  --print_detections\

#--input="/Volumes/Storage/Ring_test_examples/whole_core_examples/hidden/80xxx_short.tif" \
#--input=/Volumes/Storage/Ring_test_examples/mini_ring_examples/to_use \
#--input=/Users/miroslav.polacek/Downloads/TC_selection_of_cores/F4b_B1.png \
#--input=/Users/miroslav.polacek/Pictures/TC_selection_of_cores/cropped/ \
#--input=/Volumes/T7\ Shield/CNN_yolov8_retraining_data/Plot23_badsamplequality/chopped/00023007a_mo1973_pS1.96536799834280303030_3784.tif \
#--input=/Users/miroslav.polacek/Pictures/23_weird_image \
#--input=/Volumes/Storage/Ring_test_examples/whole_core_examples/the_real_ones/00019002a_mo4724_pS1.96536799834280303030.tif \
#--cropUpandDown=0 \
#--sliding_window_overlap=0.5 \
#--cracks \