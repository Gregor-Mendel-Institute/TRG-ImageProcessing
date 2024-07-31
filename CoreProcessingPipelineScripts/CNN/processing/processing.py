#!/usr/bin/env python
"""
Load tiff image of whole core.
Run detections for squares in a form of sliding window with certain overlap to prevent problems of having ring directly at the edge.
Run detection separately for the best model for Ring detection and the best for Crack.
Fuse all detections in one mask layer and consequently attach all of them to each other creating mask layer
of the size of original image. The detection confidence needs to be set in the config!
Export JSON and POS files.
Print the image with mask over it.
"""

#######################################################################
# Imports and set up logger
#######################################################################
import os
import sys
import cv2
import time
import argparse
import torch
from datetime import datetime
from ultralytics import YOLO
import logging

# Import custom functions
ROOT_DIR = os.path.abspath("../")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library

from functions.processing_functions import apply_mask, convert_to_binary_mask\
    , sliding_window_detection_multirow, clean_up_mask, find_centerlines, measure_contours, plot_lines, write_to_json\
    , write_to_pos, plot_contours

from functions.prepare_CVAT_annot import prepare_all_annotations
from training.retraining_container import retraining

"""
stream_h = logging.StreamHandler()
stream_h.setLevel(logging.WARNING)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
stream_h.setFormatter(formatter)
logger.addHandler(stream_h)
file_h = logging.FileHandler('test_log.log') # latter put this in output or wherever appropriate
file_h.setLevel(logging.INFO)
file_h.setFormatter(formatter)
logger.addHandler(file_h)
"""
#######################################################################
# Arguments
#######################################################################
def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Segmentation of whole core')

    ## Compulsory arguments
    parser.add_argument('--dpi', required=False,
                        help="DPI value for the image")

    parser.add_argument('--run_ID', required=False,
                        help="Run ID")

    parser.add_argument('--input', required=False,
                        metavar="/path/to/image/",
                        help="Path to image file of folder")

    parser.add_argument('--weightRing', required=False,
                        metavar="/path/to/weight/file",
                        help="Path to ring weight file")

    parser.add_argument('--output_folder', required=False,
                        metavar="/path/to/out/folder",
                        help="Path to output folder")

    ## Optional arguments
    parser.add_argument('--cracks', required=False,
                        default=False,
                        metavar="/path/to/weight/file",
                        help="Path to crack weight file")

    parser.add_argument('--cropUpandDown', required=False,
                        help="Fraction of image hight to crop away on both sides")

    parser.add_argument('--sliding_window_overlap', required=False,
                        help="Proportion of sliding frame that should overlap")

    parser.add_argument('--print_detections', required=False,
                        help="yes, if printing is desired")

    parser.add_argument('--min_mask_overlap', required=False,
                        help="Minimum of detected masks to consider good detection")

    parser.add_argument('--n_detection_rows', required=False,
                        help="Minimum of detected masks to consider good detection")

    parser.add_argument('--logfile', required=False,
                        metavar="logfile",
                        help="logfile name to put in output dir. Prepends other info (used to be 'CNN_')")

    ## Additional retrainig arguments
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/treering/dataset/",
                        help='Directory of the Treering dataset')

    parser.add_argument('--logs', required=False,
                        default="./logs",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default="./logs")')

    args = parser.parse_args()
    return args

#######################################################################
# Run detection on a forlder or images
#######################################################################
def main():
    # get the arguments
    args = get_args()

    if args.output_folder is None or not os.path.exists(args.output_folder):
        print("Compulsory argument --output_folder is missing or is not correct. Specify the path to output folder.")
        exit()
    # set up logging
    path_out = os.path.join(args.output_folder, args.run_ID)
    # Check if output dir for run_ID exists and if not create it
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    now = datetime.now()
    dt_string_name = now.strftime('D%Y%m%d_%H%M%S')  # "%Y-%m-%d_%H:%M:%S"
    run_ID = args.run_ID
    log_file_name = str(args.logfile) + run_ID + '_' + dt_string_name + '.log'
    log_file_path = os.path.join(path_out, log_file_name)

    logging.basicConfig(level=logging.INFO, filename=log_file_path,
                        format='%(asctime)s-%(name)s-%(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    logging.info(f"Output path set to: {path_out}")
    # PREPARE THE MODEL
    # Check compulsory argument
    if args.weightRing is None or not os.path.isfile(args.weightRing):
        print("Compulsory argument --weightRing is missing or is not correct. Specify the path to ring weight file.")
        logger.info("Compulsory argument --weightRing is missing or is not correct. Specify the path to ring weight file.")
        exit()
    logger.info(f"Loading weights: {args.weightRing}")
    modelRing = YOLO(args.weightRing)

    # check available devices to run model
    if torch.cuda.device_count() > 0:
        logger.info(f"{torch.cuda.device_count()} cuda devices are available")
        device_names = [torch.cuda.get_device_name(device_n) for device_n in range(torch.cuda.device_count())]

    else:
        device_names = 'CPU'

    logger.info(f"Model is running on: {device_names}")
    print(f"Model is running on: {device_names}")

    # RETRAINING
    if args.dataset is not None:
        if not os.path.exists(args.dataset):
            print("Path to --dataset does not exist.")

        print("Starting retraining mode")
        logger.info("Starting retraining mode")
        # Check and prepare annotations
        prepare_all_annotations(dataset_path=args.dataset, buffer=10, overwrite_existing=True)

        # Start retraining
        retraining(model=modelRing, dataset=args.dataset, out_path=path_out) #pass the dataset and saving location
    # DETECTION
    else:
        print("Starting inference mode")
        logger.info("Starting inference mode")
        # Check compulsory argument and print which are missing
        print('Checking compulsory arguments')
        if args.input is None or not os.path.exists(args.input):
            print("Compulsory argument --input is missing. Specify the path to image file of folder")
            logger.info("Compulsory argument --input is missing or is not correct. Specify the path to image file of folder.")
            exit()
        if args.dpi is None:
            print("Compulsory argument --dpi is missing. Specify the DPI value for the image")
            logger.info("Compulsory argument --dpi is missing. Specify the DPI value for the image")
            exit()
        if args.run_ID is None:
            print("Compulsory argument --run_ID is missing. Specify the Run ID")
            logger.info("Compulsory argument --run_ID is missing. Specify the Run ID")
            exit()

        # Create a list of already exported jsons to prevent re-running the same image
        json_l = tuple(f.replace('.json', '') for f in os.listdir(path_out) if f.endswith('.json'))

        input = args.input
        # Check pathin if its folder or file and get file list of either
        if os.path.isfile(input):
            # Get file name and dir to file
            input_l = [os.path.basename(input)]
            input_path = os.path.split(input)[0]
        elif os.path.isdir(input):
            # Get a list of files in the dir and filter hidden files
            #input_list = [f for f in os.listdir(input) if not f.startswith('.')]
            input_l = tuple(f for f in os.listdir(input) if not f.startswith('.'))
            input_path = input
        else:
            print("Input argument is neither valid file nor directory") # input or image?
            logger.info("Input argument is neither valid file nor directory")

        for f in input_l:
            supported_extensions = ('.tif', '.tiff', '.png')
            file_extension = os.path.splitext(f)[1]

            if file_extension in supported_extensions and os.path.splitext(f)[0] in json_l:
                # print image name first to keep the output consistent
                print("Processing image: {}".format(f))
                logger.info("Processing image: {}".format(f))
                print("JSON FILE FOR THIS IMAGE ALREADY EXISTS IN OUTPUT")
                logger.info("JSON FILE FOR THIS IMAGE ALREADY EXISTS IN OUTPUT")
            elif file_extension in supported_extensions and os.path.splitext(f)[0] not in json_l:
                try:
                    image_start_time = time.perf_counter()
                    print("Processing image: {}".format(f))
                    logger.info("Processing image: {}".format(f))
                    image_path = os.path.join(input_path, f)
                    im_origin = cv2.imread(image_path)
                    image_name = os.path.splitext(f)[0]  # later for saving files

                    # Define default values if they were not provided as arguments
                    if args.cropUpandDown is not None:
                        cropUpandDown = float(args.cropUpandDown)
                    else:
                        cropUpandDown = 0.17

                    if args.sliding_window_overlap is not None:
                        sliding_window_overlap = float(args.sliding_window_overlap)
                    else:
                        sliding_window_overlap = 0.75

                    if args.n_detection_rows is None or args.n_detection_rows==1:
                        detection_rows = 1
                    else:
                        detection_rows = int(args.n_detection_rows)

                    if args.cracks == 'True':
                        cracks = True
                    else:
                        cracks = False

                    if args.min_mask_overlap is not None:
                        min_mask_overlap = int(args.min_mask_overlap)
                    else:
                        min_mask_overlap = 3

                    # RUN DETECTION
                    detected_mask = sliding_window_detection_multirow(image=im_origin,
                                                            detection_rows=detection_rows,
                                                            model=modelRing,
                                                            cracks=cracks,
                                                            overlap=sliding_window_overlap,
                                                            cropUpandDown=cropUpandDown)

                    logger.info("sliding_window_detection_multirow done")
                    print("sliding_window_detection_multirow done")

                    # CLEAN UP MASKS
                    ## RINGS
                    detected_mask_rings = detected_mask[:, :, 0]
                    # print("detected_mask_rings", detected_mask_rings.shape)
                    clean_contours_rings = clean_up_mask(detected_mask_rings,
                                                         min_mask_overlap=min_mask_overlap, is_ring=True)
                    logger.info("clean_up_mask done")
                    print("clean_up_mask done")

                    ## CRACKS
                    clean_contours_cracks = None
                    if cracks is True:
                        detected_mask_cracks = detected_mask[:, :, 1]
                        print("detected_mask_cracks", detected_mask_cracks.shape)
                        clean_contours_cracks = clean_up_mask(detected_mask_cracks, is_ring=False)
                        logger.info("clean_up_mask cracks done")
                        print("clean_up_mask cracks done")

                    # FIND CENTERLINES
                    print("debug check")
                    centerlines_rings = find_centerlines(clean_contours_rings,
                                                         cut_off=0.01, y_length_threshold=im_origin.shape[0]*0.05)
                    logger.info("find_centerlines done")
                    print("find_centerlines done")
                    if centerlines_rings is None:
                        logger.info("No centerlines were detected")
                        print("No centerlines were detected")
                        centerlines = None
                        measure_points = None
                        finished = False

                    else:
                        # MEASURE RING DISTANCES
                        # if statement to prevent crushing in case centerlines_rings contains only one centerline
                        if centerlines_rings.geom_type == 'LineString':
                            logger.info("centerlines_rings contains only one centerline for this image")
                            print("centerlines_rings contains only one centerline for this image")
                            centerlines = centerlines_rings  # for visualisation of the result
                            measure_points = None
                            finished = False

                        elif centerlines_rings.geom_type == 'MultiLineString':
                            centerlines, measure_points, cutting_point = measure_contours(centerlines_rings, detected_mask_rings)
                            logger.info("measure_contours done")
                            print("measure_contours done")

                            # WRITE RING MEASUREMENTS
                            write_to_json(image_name=f, cutting_point=cutting_point, run_ID=run_ID,
                                          path_out=path_out, centerlines_rings=centerlines_rings,
                                          clean_contours_rings=clean_contours_rings,
                                          clean_contours_cracks=clean_contours_cracks)
                            logger.info("write_to_json done")

                            DPI = float(args.dpi)
                            write_to_pos(centerlines, measure_points, image_name, f, DPI, path_out)
                            finished = True

                    # PRINT DETECTED IMAGES
                    if args.print_detections == "yes":
                        # Plotting lines is mostly for debugging
                        masked_image = im_origin.copy()
                        print("masked_image.dtype", masked_image.dtype)
                        masked_image = apply_mask(masked_image, detected_mask_rings, alpha=0.2)

                        if cracks is True:
                            masked_image = apply_mask(masked_image, detected_mask_cracks, alpha=0.3)

                        plot_lines(masked_image, centerlines, measure_points,
                                    image_name, path_out)
                        logger.info("plot_lines done")

                    if finished:
                        logger.info("IMAGE FINISHED")
                        print("IMAGE FINISHED")
                    else:
                        logger.info("IMAGE WAS NOT FINISHED")
                        print("IMAGE WAS NOT FINISHED")

                    image_finished_time = time.perf_counter()
                    image_run_time = image_finished_time - image_start_time
                    logger.info(f"Image run time: {image_run_time} s")

                except Exception as e:
                    logger.exception(e)
                    logger.info("IMAGE WAS NOT FINISHED")
                    print(e)
                    print("IMAGE WAS NOT FINISHED")

if __name__ == '__main__':
    main()
