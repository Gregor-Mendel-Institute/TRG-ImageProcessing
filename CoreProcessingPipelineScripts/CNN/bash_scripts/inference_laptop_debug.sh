
time python ../postprocessing/postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=Squares \
  --input=/Volumes/T7\ Shield/CNN_yolov8_retraining_data/OurImages_raw/chopped/00019005b_mo4577_pS1.96536799834280303030_0.tif \
  --weightRing=../weights/best10pxlowerlrf.pt \
  --output_folder=../output \
  --cracks=True \
  --print_detections=yes

#--input=/Volumes/Storage/Ring_test_examples/mini_ring_examples/to_use \
#--input=/Users/miroslav.polacek/Downloads/TC_selection_of_cores/F4b_B1.png \
#--input=/Users/miroslav.polacek/Pictures/TC_selection_of_cores/cropped/ \
#--cropUpandDown=0 \