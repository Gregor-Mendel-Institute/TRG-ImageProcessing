time python ../processing/processing.py \
  --training_data=/Users/miroslav/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/testing_and_development/sample_dataset_mini_testing \
  --weights=../weights/best10px1000eAugEnlargedDataset.pt \
  --output_folder=/Users/miroslav/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/output \
  --epochs=1 \
  --debug \
  --generate_annotations \
  --annot_buffer=10 \
  --run_ID=Train_debug_5

#--generate_annotations \
#--annot_buffer=0 \
#--training_data= \
#--weights=../weights/best10px1000eAugEnlargedDataset.pt \
#--weights="/Users/miroslav/Downloads/yolov12x-seg.pt" \