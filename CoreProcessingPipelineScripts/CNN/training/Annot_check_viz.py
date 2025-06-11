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

from functions.prepare_CVAT_annot import prepare_all_annotations
from functions.processing_functions import check_annot_dataset

# prepare variables
#DATASET_PATH = "/Users/miroslav/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/"
DATASET_PATH = "/Users/miroslav/Documents/Timon_annot/Timon_all_annot_squares/Timon_data_training"

# testing on sample dataset
start_time = time.perf_counter()
prepare_all_annotations(dataset_path=DATASET_PATH, buffer=10, overwrite_existing=True)
check_annot_dataset(DATASET_PATH)
finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}") # on mac the original:  32-34s


