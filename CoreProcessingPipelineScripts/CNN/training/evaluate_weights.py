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
import datetime

# os.chdir("/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/functions")
# Import custom functions
ROOT_DIR = os.path.abspath("../")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library

from functions.processing_functions import (apply_mask, convert_to_binary_mask, load_annot,
                                            sliding_window_detection_multirow, clean_up_mask)

######################### FUNCTIONS #############################################
def load_annot_Polygons(yolo_annot_file, im_size, n_classes, cropUpandDown):
    annot = load_annot(yolo_annot_file, im_size)
    if cropUpandDown > 0:
        to_crop = int(im_size[0] * cropUpandDown)
        crop_box = shapely.geometry.box(0, to_crop, im_size[1], im_size[0] - to_crop)  # (minx, miny, maxx, maxy)
    else:
        crop_box = shapely.geometry.box(0, 0, im_size[1], im_size[0])  # (minx, miny, maxx, maxy)

    #print("crop_box bounds", crop_box.bounds)
    polys = [[] for _ in range(n_classes)]
    for an, c_id in zip(annot[0], annot[1]):
        polys[int(c_id)].append(shapely.geometry.Polygon(an).intersection(crop_box))

    #print("at the end", polys)
    #print("first ring bounds", polys[0][0].bounds)
    return polys

def get_detection_polys(image, model, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap):
    detected_mask = sliding_window_detection_multirow(image=image,
                                                      detection_rows=detection_rows,
                                                      model=model,
                                                      cracks=True,
                                                      overlap=sliding_window_overlap,
                                                      cropUpandDown=cropUpandDown)
    # CLEAN UP MASKS
    ## RINGS
    detected_mask_rings = detected_mask[:, :, 0]
    # print("detected_mask_rings", detected_mask_rings.shape)
    clean_contours_rings = clean_up_mask(detected_mask_rings,
                                         min_mask_overlap=min_mask_overlap, is_ring=True)

    ## CRACKS
    detected_mask_cracks = detected_mask[:, :, 1]
    clean_contours_cracks = clean_up_mask(detected_mask_cracks, is_ring=False)
    return (clean_contours_rings, clean_contours_cracks)

def _get_metrics(poly_d, poly_t, IoU_thresholds):
    # Calculate metrics per image per class
    # poly_d and poly_t are detected and truth shapely polygons
    # Precision as correctly detected/all detected
    # Recall as correctly detected/all real (ground truth) rings
    #print("poly_d", poly_d)
    #print("poly_t", poly_t)
    # ADD COMPREHENSION TO filter ONLY VALID POLYGONS
    #poly_t_v = [pT for pT in poly_t if shapely.is_valid(pT)] # just to see but remove, does not make sense they should be good
    #poly_d_v = [pD for pD in poly_d if shapely.is_valid_reason(pD)]
    #print(f'poly_t: {len(poly_t)}')
    """
    for pD in poly_d:
        print("area", pD.area)
        reason = shapely.is_valid_reason(pD)
        print("reason", reason)
        if 'Ring Self-intersection' in reason:
            x, y = pD.exterior.coords.xy
            plt.plot(x,y)
            plt.show()

    #print(f'poly_d: {len(poly_d)} poly_d_v: {len(poly_d_v)}')
    """
    if len(poly_t) == 0:
        NANs = np.repeat(np.nan, len(IoU_thresholds))
        P, R, IoU = NANs, NANs, NANs
    elif len(poly_d) == 0 and len(poly_t) != 0:
        zeros = np.repeat(0, len(IoU_thresholds))
        P, R, IoU = zeros, zeros, np.repeat(np.nan, len(IoU_thresholds))
    else:
        IoU_list = [max((pT.intersection(pD).area / pT.union(pD).area for pD in poly_d))
                         for pT in poly_t]

        TPs = np.array([len(np.where(IoU_list > IoU_threshold)[0]) for IoU_threshold in IoU_thresholds])
        #print("TPs", TPs)
        P = TPs / len(poly_d)
        R = TPs / len(poly_t)
        #print("IoU_list", IoU_list)
        IoU = [np.mean(IoU_list)] + np.repeat(np.nan, len(IoU_thresholds)-1).tolist() # make them same dimension to convert everything in np.array
    #print("IoU", IoU)
    #print("len IoU", len(IoU))
    return P, R, IoU

def eval_image(image, model, yolo_annot_file, im_size, n_classes, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds):
    # get ground truth as polygons
    # GT and D are polygons by category. Ring in position 0 and crack in 1
    polys_gt = load_annot_Polygons(yolo_annot_file, im_size, n_classes, cropUpandDown)
    #print("polys_gt", polys_gt)
    # run detection and prepare the polygons
    polys_d = get_detection_polys(image, model, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap)
    # in that order are also the their metrics
    P, R, IoU = [], [], []
    for p_gt, p_d in zip(polys_gt, polys_d):
        Pt, Rt, IoUt = _get_metrics(poly_d=p_d, poly_t=p_gt, IoU_thresholds=IoU_thresholds)
        #print("len Pt, Rt, IoUt", len(Pt), len(Rt), len(IoUt))
        P.append(Pt)
        R.append(Rt)
        IoU.append(IoUt)

    return (P, R, IoU)

def eval_dataset(data, model, n_classes, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds):
    supported_extensions = ('.tif', '.tiff', '.png')
    im_list = (i for i in os.listdir(data) if os.path.splitext(i)[1] in supported_extensions and not i.startswith("."))
    results = []
    # im_name = "33627_201908231505-01(4)_8037a.tif"
    # im_name = "2019102817-01(12)_00015058a33_m01.tif" # many empty polygons loaded
    # im_name = "20115_00041007a_0_pSX1.965424714300121_pSY1.9655438706947042.tif"
    for im_name in im_list:
        ## load image to extract the im size and other values
        print("evaluating image", im_name)
        im_path = os.path.join(data, im_name)
        im = cv2.imread(im_path)
        im_size = im.shape[:2]
        im_name_no_ext = os.path.splitext(im_name)[0]
        yolo_annot_file = os.path.join(data, im_name_no_ext + ".txt")

        ## load annotations by image name as shapely polygon per class
        im_res = eval_image(im, model, yolo_annot_file, im_size, n_classes, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds)
        results.append(im_res)
    out_array = np.nanmean(np.array(results), axis=0)  # average along the images
    return out_array
#################################################################################
# SET VARIABLES
#DATASET = "/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/"
DATASET = "/groups/swarts/user/miroslav.polacek/UpdatedTRGDataset10px"
#weights = "/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/weights/best10px1000eAugEnlargedDataset.pt"
weights ="/groups/swarts/user/miroslav.polacek/TRG-ImplementYolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/weights/best10px1000eAugEnlargedDataset.pt"

data_yaml = os.path.join(DATASET, "data.yaml")
OUTPUT_PATH = os.path.join(ROOT_DIR, "output")
evals_dir = os.path.join(OUTPUT_PATH, "evals")
test_name = "best10px1000eAugEnlargedDataset" # later will be derived in function
output_folder = os.path.join(evals_dir, test_name)
# Check if output dir for run_ID exists and if not create it
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
# more parameters
detection_rows = 1
sliding_window_overlap = 0.5
# cropUpandDown = 0.2
min_mask_overlap = 3
IoU_thresholds = np.arange(0.5,1,0.05)
n_classes = 2
"""
# SET UP LOGGER
now = datetime.now()
dt_string_name = now.strftime('D%Y%m%d_%H%M%S')  # "%Y-%m-%d_%H:%M:%S"
log_file_name = 'Eval_log' + '_' + dt_string_name + '.log'
log_file_path = os.path.join(output_folder, log_file_name)

logging.basicConfig(level=logging.INFO, filename=log_file_path,
                    format='%(asctime)s-%(name)s-%(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
if args.debug == 'True':
    logging.getLogger().setLevel(logging.DEBUG)
"""
# RUN GENERAL YOLO EVALUATION
model = YOLO(weights)
res_yolo = model.val(data = data_yaml, project=evals_dir, name=test_name)

## search all the images in the
data = os.path.join(DATASET, "val")

###### TEST DIFFERENT CROP ######
cropUpandDown_tuple = (0.2, 0.17, 0.10, 0)
for cropUpandDown in cropUpandDown_tuple:
    res_arr = eval_dataset(data, model, n_classes, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds)
    out_file_results = os.path.join(output_folder, "Res_array" + str(cropUpandDown) + ".npy")
    np.save(out_file_results, res_arr)

# nanmean_across_IoU_thresholds = np.nanmean(nanmean_ar, axis=2)
load_file_results = os.path.join(output_folder, "Res_array0.2.npy")
test = np.load(load_file_results)

###### Debug stuff delete #######
for geom in pv.geoms:
    x, y = geom.exterior.xy
    plt.plot(x, y)
plt.show()