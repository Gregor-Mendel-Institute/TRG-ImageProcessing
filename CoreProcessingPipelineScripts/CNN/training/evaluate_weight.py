import os
import sys
import argparse
import torch
from datetime import datetime
from ultralytics import YOLO
import numpy as np
import logging
import platform


# Import custom functions
ROOT_DIR = os.path.abspath("../")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from functions.training_functions import eval_dataset, plot_results

# set the argsparse
def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate dataset')

    ## Compulsory arguments
    parser.add_argument('--run_ID', required=False,
                        help="Run ID")

    parser.add_argument('--input', required=False,
                        metavar="/path/to/image/",
                        help="Path to image file of folder")

    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights/file",
                        help="Path to weights file")

    parser.add_argument('--output_folder', required=False,
                        metavar="/path/to/out/folder",
                        help="Path to output folder")

    ## Optional arguments
    parser.add_argument('--cropUpandDown', required=False,
                        default=0.17,
                        type=float,
                        help="Fraction of image hight to crop away on both sides")

    parser.add_argument('--sliding_window_overlap', required=False,
                        default=0.75,
                        type=float,
                        help="Proportion of sliding frame that should overlap")

    parser.add_argument('--print_detections', required=False,
                        default=False,
                        help="True, if printing is desired")

    parser.add_argument('--min_mask_overlap', required=False,
                        default=3,
                        type=int,
                        help="Minimum of detected masks to consider good detection")

    parser.add_argument('--n_detection_rows', required=False,
                        default=1,
                        type=int,
                        help="Minimum of detected masks to consider good detection")

    parser.add_argument('--logs', required=False,
                        default="./logs",
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default="./logs")')

    parser.add_argument('--logfile', required=False,
                        metavar="logfile",
                        help="logfile name to put in output dir. Prepends other info (used to be 'CNN_')")

    parser.add_argument('--debug', required=False,
                        default=False,
                        help="True will set logging level to debug")

    ## Additional retrainig arguments
    parser.add_argument('--training_data', required=False,
                        metavar="/path/to/training/dataset/",
                        help='Directory of the training dataset')

    parser.add_argument('--generate_annotations', required=False,
                        default=False,
                        help='If you wish to generate annotations, or overwrite existing')

    parser.add_argument('--annot_buffer', required=False,
                        default=10,
                        type=int,
                        help='By how much should the boundary line be buffered for training')

    parser.add_argument('--epochs', required=False,
                        default=1500,
                        type=int,
                        help='Number of training iteration to run the model')

    args = parser.parse_args()
    return args
# main
def main():
# get the arguments
    args = get_args()

## set up logger
# first need output folder for logging file
    if args.output_folder is None or not os.path.exists(args.output_folder):
        print(f"Compulsory argument --output_folder is missing or the path {args.output_folder} does not exist.")
        exit()

    path_out = os.path.join(args.output_folder, "evals")

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
    if args.debug == 'True':
        logging.getLogger().setLevel(logging.DEBUG)

    logging.info(f"Output path set to: {path_out}")
    # Report os and python
    logger.debug(f"OS specs: {platform.platform()}")
    logger.debug(f"Python version: {platform.python_version()}")

    # PREPARE THE MODEL
    # Check compulsory argument
    logger.debug(f"args.weights: {args.weights}")
    if args.weights:
        if args.weights.endswith('.pt'):
            pass
        elif not os.path.isfile(args.weights):
            print(f"Compulsory argument --weights path {args.weights} does not exist.")
            logger.warning(f"Compulsory argument --weights is missing or the path {args.weights} does not exist.")
            exit()
    else:
        print(f"Compulsory argument --weights is missing.")
        logger.warning(f"Compulsory argument --weights is missing.")
        exit()
    logger.info(f"Loading weights: {args.weights}")
    model = YOLO(args.weights)

    # check available devices to run model
    if torch.cuda.device_count() > 0:
        logger.debug(f"{torch.cuda.device_count()} cuda devices are available")
        device_names = [torch.cuda.get_device_name(device_n) for device_n in range(torch.cuda.device_count())]

    else:
        device_names = 'CPU'

    logger.info(f"Model is running on: {device_names}")
    print(f"Model is running on: {device_names}")
    ## prepare variables and paths
    ## prepare input variables
    data = os.path.join(args.training_data, "val")
    n_classes = 2
    IoU_thresholds = np.arange(0.5, 1, 0.05)
    # Create output path
    run_name = args.run_ID
    res_out_path = os.path.join(path_out, run_name)
    # make if does not exist
    if not os.path.isdir(res_out_path):
        os.makedirs(res_out_path)

    ## run eval
    res_arr = eval_dataset(data, model, n_classes, args.n_detection_rows, args.sliding_window_overlap,
                           args.cropUpandDown, args.min_mask_overlap, IoU_thresholds)
    #save plots and results
    summary_out = np.nanmean(res_arr, axis=2)
    csv_file_out = os.path.join(res_out_path, run_name.replace(".pt", ".csv"))
    summary_out.tofile(csv_file_out, sep=',')  # , format='%10.5f')
    out_file_plot = os.path.join(res_out_path, run_name.replace(".pt", ".png"))
    plot_results(res_arr, IoU_thresholds, out_file_plot)


if __name__ == '__main__':
    main()