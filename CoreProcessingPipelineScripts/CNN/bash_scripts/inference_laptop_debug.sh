
time python ../postprocessing/postprocessingCracksRings.py \
  --dpi=13039 \
  --run_ID=test_Timon_noCrop \
  --input=/Users/miroslav.polacek/Downloads/TC_selection_of_cores/F4b_B1.png \
  --weightRing=/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/weights/best10pxlowerlrf.pt \
  --output_folder=../output \
  --cracks=True \
  --print_detections=yes

#--input=/Volumes/Storage/Ring_test_examples/mini_ring_examples/to_use \
#--cropUpandDown=0 \