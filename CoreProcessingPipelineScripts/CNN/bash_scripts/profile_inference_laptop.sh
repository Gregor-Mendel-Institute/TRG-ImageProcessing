
time python -m cProfile ../processing/processing.py \
  --dpi=13039 \
  --run_ID=Profiling \
  --input=/Users/miroslav.polacek/Pictures/23_weird_image \
  --weightRing=../weights/best10px1500eAugEnlargedDataset.pt \
  --output_folder=../output \
  --sliding_window_overlap=0.5 \
  --cracks=True \
  --debug=True \
  --print_detections=yes\
  > cprofile.txt

#--input=/Volumes/Storage/Ring_test_examples/mini_ring_examples/to_use \
#--input=/Users/miroslav.polacek/Downloads/TC_selection_of_cores/F4b_B1.png \
#--input=/Users/miroslav.polacek/Pictures/TC_selection_of_cores/cropped/ \
#--input=/Volumes/T7\ Shield/CNN_yolov8_retraining_data/Plot23_badsamplequality/chopped/00023007a_mo1973_pS1.96536799834280303030_3784.tif \
#--input=/Users/miroslav.polacek/Pictures/23_weird_image \
#--cropUpandDown=0 \