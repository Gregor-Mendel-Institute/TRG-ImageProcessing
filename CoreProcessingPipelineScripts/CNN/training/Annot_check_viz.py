"""
The function should take all the images in given folder, print pngs with annotations in "annot_check" subdirectory, on
the way write in a log summary of annotations: how many images, how many rings and how many cracks, list images that
do not have cracks or rings and images that do not have anny annotations save a text file with all of this in
the subdirectory as well.
"""
import os
import time

# Import Mask RCNN
ROOT_DIR = os.path.abspath('./CoreProcessingPipelineScripts/CNN/') # to run in Pycharm
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from functions.processing_functions import check_annot_dataset

# testing on sample dataset
start_time = time.perf_counter()
dataset = "/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/"
check_annot_dataset(dataset)
finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}") # on mac the original:  32-34s

# run on my dataset
dataset = "/Volumes/T7 Shield/TRG_RingCrack_3rd_trainingdataset"
check_annot_dataset(dataset)

