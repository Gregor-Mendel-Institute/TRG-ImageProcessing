
time python ../processing/processing.py \
  --dpi=13039 \
  --run_ID=Test_line_fix_08  \
  --input=/Users/miroslav/Documents/Pictures/Test_cores_from_speed_test/00014006b_core_id7_bc4_obj0_pid2_tN2_pS0.6485714396165350_pSX1.9653784325736006_pSY1.9658008396911357_crop.tif \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=../output \
  --cropUpandDown=0 \
  --sliding_window_overlap=0.75 \
  --cracks=True \
  --debug=True \
  --print_detections=True\

#--input=/Volumes/Storage/Ring_test_examples/mini_ring_examples/to_use \
#--input=/Users/miroslav.polacek/Downloads/TC_selection_of_cores/F4b_B1.png \
#--input=/Users/miroslav.polacek/Pictures/TC_selection_of_cores/cropped/ \
#--input=/Volumes/T7\ Shield/CNN_yolov8_retraining_data/Plot23_badsamplequality/chopped/00023007a_mo1973_pS1.96536799834280303030_3784.tif \
#--input=/Users/miroslav.polacek/Pictures/23_weird_image \
#--input=/Volumes/Storage/Ring_test_examples/whole_core_examples/the_real_ones/00019002a_mo4724_pS1.96536799834280303030.tif \
#--cropUpandDown=0 \
#--sliding_window_overlap=0.5 \