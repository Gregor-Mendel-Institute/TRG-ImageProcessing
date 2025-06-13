# Evaluate weights on validation (or whatever annotated dataset)
# Would be good to have the following metrics per image per class: IoU, precision, recall, mAP and detection time.
# Per image values can be averaged but potentially the worst performing images can be highlighted, e.g. make folder
# with evaluation and printed 10 worst and 10 best images with detections and ground truth
# Then I can use it also to test hyperparameter such crop up down
# Output a graph with precision and recall over IoU thresholds per every class
# RUN EVALUATION WITH DATA PRE AND POST PROCESSING
# prepare annotations in COCO format
# Function at image level output precision (from 0.5 to 0.95 by 0.5 iou), recall(from 0.5 to 0.95 by 0.5 iou), IoU
# and for each category

import os
from ultralytics import YOLO
import numpy as np
import shapely
import cv2
import sys
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import argparse

# set up logger
logger = logging.getLogger(__name__)
# os.chdir("/Users/miroslav/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/functions")
# Import custom functions
ROOT_DIR = os.path.abspath("../")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR) # To find local version of the library

# Load training functions
from functions.training_functions import evaluate_training

######################### ARGS #################################################
def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Segmentation of whole core')

    parser.add_argument('--training_data', required=False,
                        metavar="/path/to/training/dataset/",
                        help='Directory of the training dataset')

    parser.add_argument('--run_ID', required=False,
                        help="Run ID")

    parser.add_argument('--n_detection_rows', required=False,
                        default=1,
                        type=int,
                        help="Minimum of detected masks to consider good detection")

    parser.add_argument('--sliding_window_overlap', required=False,
                        default=0.75,
                        type=float,
                        help="Proportion of sliding frame that should overlap")

    parser.add_argument('--cropUpandDown', required=False,
                        default=0.17,
                        type=float,
                        help="Fraction of image hight to crop away on both sides")

    parser.add_argument('--min_mask_overlap', required=False,
                        default=3,
                        type=int,
                        help="Minimum of detected masks to consider good detection")

    parser.add_argument('--output_folder', required=False,
                        metavar="/path/to/out/folder",
                        help="Path to output folder")

    parser.add_argument('--debug', required=False,
                        default=False,
                        type=bool,
                        help="True will set logging level to debug")

    args = parser.parse_args()
    return args

######################### FUNCTIONS #############################################

#################################################################################
def main():
    args = get_args()

    path_out = os.path.join(args.output_folder, "retraining")
    # Check if output dir for run_ID exists and if not create it
    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    # SET UP LOGGER
    now = datetime.now()
    dt_string_name = now.strftime('D%Y%m%d_%H%M%S')  # "%Y-%m-%d_%H:%M:%S"
    log_file_name = 'Eval_log' + '_' + dt_string_name + '.log'
    log_file_path = os.path.join(path_out, log_file_name)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file_path)],
                        format='%(asctime)s-%(name)s-%(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    if args.debug == 'True':  # args.debug == 'True'
        logging.getLogger().setLevel(logging.DEBUG)

    # Evaluate trained weights
    evaluate_training(dataset_path=args.training_data, out_path=path_out, name=args.run_ID,
                      detection_rows=args.n_detection_rows, sliding_window_overlap=args.sliding_window_overlap,
                      cropUpandDown=args.cropUpandDown, min_mask_overlap=args.min_mask_overlap)

if __name__ == '__main__':
    main()