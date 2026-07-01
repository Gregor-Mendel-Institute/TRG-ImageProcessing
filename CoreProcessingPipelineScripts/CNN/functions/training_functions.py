"""
Prepares annotations from .xml files from CVAT_annotations folder and distributes them based on the image names in
train and val folders.
In the CVAT annotation file the ring boundary has to be a polyline RingBndy and crack as polygone labeled CrackPoly.
"""

############# IMPORTS #################
import os
import sys
import copy
from lxml import etree
import shapely
from shapely.geometry import LineString, Polygon, box
from ultralytics import YOLO
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
#from datetime import datetime
import numpy as np
logger = logging.getLogger(__name__)

# Import custom functions
ROOT_DIR = os.path.abspath("../")
logger.debug("ROOT_DIR: %s", ROOT_DIR)
sys.path.append(ROOT_DIR) # To find local version of the library

from functions.processing_functions import (sliding_window_detection_multirow, clean_up_mask, log_and_print)

#########################################################
# FUNCTIONS CVAT annotations
#########################################################
#######################################################################
# Plot contours
#######################################################################
def plot_contours(image, contours, file_name, path_out, labels=None):
    # plot image with extracted contours to facilitate debuging
    logger.info("plot_contours START")
    image_copy = copy.copy(image)
    contours = tuple(contours)
    if labels:
        color = [(0, 255, 0), (255, 0, 0)]
        for l in labels:
            contours_r = (contour for i, contour in enumerate(contours) if labels[i] == l)
            for contour in contours_r:
                cv2.drawContours(image_copy, [contour], -1, color[int(l)], 2)

    else:
        for contour in contours:
            cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2)

    logger.info("Plotting output as png")
    export_path = os.path.join(path_out, 'pngs')

    # checks and creates if does not exists
    os.makedirs(export_path, exist_ok=True)

    f = os.path.splitext(file_name)[0] + '.png' # can not use replace as original extension can be .tif, or .tiff or .png...
    # test simpler way of saving
    cv2.imwrite(os.path.join(export_path, f), image_copy)

    logger.info("plot_contours FINISH")
#######################################################################
# Extract annotations from yolov8 format text files
#######################################################################
def load_annot(annot_path, im_size):
    # annot_path is path to annotation txt file of yolov8 format
    # im_size is output of .shape[:2] method e.g. (im_height, im_width)
    # output contours are in shape acceptable for cv2 contours
    # load annotations
    logger.info("load_annot START")
    labels, contours = [], []
    with open(annot_path, "r") as f:
        for line in f:
            if len(line) < 3:  # it has to be much more to be valid points but in my case its sometimes a space in a row
                continue
            splt_parts = line.split()
            label, annot_list = splt_parts[0], splt_parts[1:]
            labels.append(label)
            annot_list = list(map(float, annot_list))  # convert x,y from string to a value

            # the third might avoid for loop entirely and rely on numpy only
            coords = np.asarray(annot_list, dtype=np.float32).reshape(-1, 2)

            coords[:, 0] *= im_size[1]
            coords[:, 1] *= im_size[0]

            contours.append(coords.astype(np.int32))

    logger.info("load_annot FINISH")
    return contours, labels
##########################################################################################
# Function to print annotations as a png files plus save txt with some summary information
##########################################################################################
def check_annot_folder(dataset_path, folder):
    logger.info("check_annot_folder START")
    out_path = os.path.join(dataset_path, "annot_check", folder)
    folder_path = os.path.join(dataset_path, folder)
    log_and_print(f"folder_path: {folder_path}", logger, "info")

    os.makedirs(out_path, exist_ok=True)

    supported_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    im_list = (f for f in os.listdir(folder_path) if f.endswith(supported_extensions) and not f.startswith('.'))
    no_annot_file, no_annot_im = [], []
    ok_im_count, im_with_ring, im_with_crack, im_with_both = 0, 0, 0, 0
    ring_n, crack_n = 0, 0

    for im_name in im_list:
        log_and_print(f"im_name: {im_name}", logger, "info")
        im_path = os.path.join(folder_path, im_name)
        im = cv2.imread(im_path)
        im_size = im.shape
        #annot_path = im_path.replace(".tif", ".txt")
        annot_path = os.path.splitext(im_path)[0] + '.txt'
        log_and_print(f"annot_path: {annot_path}", logger, "info")
        #print("annot_path", annot_path)
        if not os.path.exists(annot_path):
            no_annot_file.append(im_name)
            log_and_print(f"Annot file for image {im_name} does not exist", logger, "warning")
            #print(f"Annot file for image {im_name} does not exist")
            continue
        contours, labels = load_annot(annot_path, im_size)
        if len(labels) == 0:
            no_annot_im.append(im_name)
            log_and_print(f"Image {im_name} has no annotations", logger, "warning")
            #print(f"Image {im_name} has no annotations")
            continue

        ring_n += labels.count('0')
        crack_n += labels.count('1')

        ok_im_count += 1
        label_set = set(labels)
        has_ring = '0' in label_set
        has_crack = '1' in label_set

        if has_ring and has_crack:
            im_with_both += 1
        elif has_ring:
            im_with_ring += 1
        elif has_crack:
            im_with_crack += 1
        """
        if '0' in set_labels and '1' in set_labels:
            im_with_both += 1
        elif '0' in set_labels and '1' not in set_labels:
            im_with_ring += 1
        elif '0' not in set_labels and '1' in set_labels:
            im_with_crack += 1
        """
        plot_contours(image=im, contours=contours, labels=labels, file_name=im_name, path_out=out_path)

    annot_info_file = os.path.join(out_path, "annot_info.txt")
    #log_and_print(f"ring_n: {ring_n}", "info")
    #log_and_print(f"crack_n: {crack_n}", "info")

    with open(annot_info_file, 'w') as f:
        f.write(f'Folder: {folder_path} \n'
                f'Ok images with annotations: {ok_im_count} \n'
                f'Data contain {ring_n} rings and {crack_n} cracks.\n'
                f'Images with only rings: {im_with_ring} only cracks: {im_with_crack} and both: {im_with_both} \n'
                f'Images without annotations {no_annot_im} \n'
                f'Images without annotation file {no_annot_file}')

    logger.info("check_annot_folder FINISH")
#######################################################################
# Run annot check on val and train folders of dataset
#######################################################################
def check_annot_dataset(dataset_path):
    logger.info("check_annot_dataset START")
    for folder in ("train", "val"):
        check_annot_folder(dataset_path, folder)
    logger.info("check_annot_dataset FINISH")

def collect_annotations(CVAT_folder):
    # extracts individual image annotations dictionaries from all xml files and put them in one list
    logger.debug("collect_annotations START")
    #CVAT_annot_file_list = [f for f in os.listdir(CVAT_folder) if not f.startswith('.')]  # ignore hidden files
    xml_files = (f for f in os.listdir(CVAT_folder) if f.endswith(".xml") and not f.startswith("."))

    # make sure they are xml files
    all_image_xml_list = []
    for xml_name in xml_files:
        # load xml file
        log_and_print(f"Loading annotation file: {xml_name}", logger, "info")

        with open(os.path.join(CVAT_folder, xml_name)) as file:
            try:
                root = etree.parse(file).getroot()
            except etree.XMLSyntaxError as e:
                logger.warning(f"Failed to parse {xml_name}: {e}")
                continue

        # this works better than a loop
        all_image_xml_list.extend(root.findall(".//image"))

    logger.debug("collect_annotations FINISH")
    return all_image_xml_list

def polylinetopolygon(polyline_str, width, height, buffer=10):
    # takes polyline string form CVAT xml.
    # Using shapely package it transforms line into polygon
    # Output x and y coords of polygone.
    # width and height of the image ensure that polygone is not exceeding image
    # buffer=0 will export only the lines coordinates for rings
    logger.debug("polylinetopolygon START")
    #coords = list((x, y) for point.split(",") in points)
    xy_coords = list(tuple(map(float, (point.split(",")))) for point in polyline_str.split(";"))
    polyline = LineString(xy_coords)
    if not polyline_str: # In case polyline will end up empty
        logger.warning("Empty polyline_str")
        return (), ()
    im_box = box(1, 1, int(width)-1, int(height)-1) # to be sure I crop it one pixel inside the image
    if buffer>0:
        polygon_ring = polyline.buffer(buffer)
        polygon_clean = polygon_ring.intersection(im_box) # in theory can be split to different geometry by intersection but practically impossible
        x, y = polygon_clean.exterior.coords.xy
    else:
        x, y = polyline.coords.xy
    logger.debug("polylinetopolygon FINISH")
    return tuple(x), tuple(y)

def preparepolygon(polygon_str):
    # Output x and y coords of polygone.
    logger.debug("preparepolygon START")
    if not polygon_str: # In case polyline will end up empty
        logger.warning("Empty polygon_str")
        return (), ()
    #points = polygon_str.split(";")
    xy_coords = list(tuple(map(float, (point.split(",")))) for point in polygon_str.split(";"))
    #polygon = Polygon(xy_coords)
    x, y = Polygon(xy_coords).exterior.coords.xy
    logger.debug("preparepolygon FINISH")
    return tuple(x), tuple(y)

def prepare_annotations(dataset_path, annot_list, buffer=10, overwrite_existing=False):
    # annot_list is a generator from collect_annotations
    logger.debug("prepare_annotations START")
    dir_list = os.listdir(dataset_path)
    supported_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    im_set = {f for f in dir_list if f.endswith(supported_extensions) and not f.startswith('.')}

    for i in annot_list:
        # get image name to control the loop
        image_name = i.get("name")
        annot_txt_file = os.path.join(dataset_path, os.path.splitext(image_name)[0] + ".txt")
        if image_name not in im_set:
            continue
        if os.path.isfile(annot_txt_file) and not overwrite_existing:
            continue

        logger.debug("Preparing annotations for image: %s", image_name)
        im_width = int(i.get("width"))
        im_height = int(i.get("height"))
        # prepare inverts of size so I can use cheaper multiplication
        inv_width = 1.0 / im_width
        inv_height = 1.0 / im_height

        # precompute processors for both polyline or polygon
        processors = {
            "polyline": lambda p: polylinetopolygon(polyline_str=p, width=im_width, height=im_height, buffer=buffer),
            "polygon": lambda p: preparepolygon(p)}

        # polylines are Ring annotations and polygons Cracks
        with open(annot_txt_file, "w+") as f:
            for poly_tag_type in ("polygon", "polyline"): #"polyline",
                logger.debug("Loop with poly_tag_type: %s", poly_tag_type)
                for poly_tag in i.iter(poly_tag_type):
                    logger.debug("Loop with poly_tag_type: %s", poly_tag_type)
                    logger.debug("poly_tag: %s", poly_tag)

                    points = poly_tag.get("points")
                    logger.debug("points: %s", points)
                    if not points:
                        logger.warning("Missing points in %s", image_name)
                        continue

                    all_points_x, all_points_y = processors[poly_tag_type](points)


                    # normalize the coordinates by image size
                    #all_points_x_norm = [x * inv_width for x in all_points_x]
                    #all_points_y_norm = [y * inv_height for y in all_points_y]

                    # create and save the line of this ring Ring is 0 crack is 1
                    annot_label = "1 " if poly_tag_type == "polygon" else "0 "


                    line = annot_label + " ".join(f"{x * inv_width} {y * inv_height}"
                                                  for x, y in zip(all_points_x, all_points_y)) + "\n"

                    logger.debug("Annot line %s", line)

                    f.write(line)

    logger.debug("prepare_annotations FINISH")

######## MAIN ############################
def prepare_all_annotations(dataset_path, buffer=10, overwrite_existing=True):
    logger.debug("prepare_all_annotations START")
    all_annot_list = collect_annotations(CVAT_folder=os.path.join(dataset_path, 'CVAT_annotations'))

    for subset in ('train', 'val'):
        logger.debug("subset: %s", subset)
        prepare_annotations(dataset_path=os.path.join(dataset_path, subset), annot_list=all_annot_list,
                            buffer=buffer, overwrite_existing=overwrite_existing)
    logger.debug("prepare_all_annotations FINISH")
###### TESTING ##########
# DATASET_PATH = '/Users/miroslav.polacek/Github/TRG_YOLOv8_try/training/sample_dataset' # for development
# DATASET_PATH = '/Volumes/swarts/user/miroslav.polacek/FullSpruceDatasetWithCVAT5pxBuffer'
# prepare_all_annotations(dataset_path=DATASET_PATH, buffer=5, overwrite_existing=True)

#########################################################
# FUNCTIONS training
#########################################################
def create_data_yaml(dataset_path):
    logger.debug("create_data_yaml START")
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    logger.debug("path: %s", data_yaml_path)
    with open(data_yaml_path, 'w') as f:
        f.write(f'path: {os.path.abspath(dataset_path)}\n'
                f'train: train\n'
                f'val: val\n'
                f'names:\n'
                f'  0: ring\n'
                f'  1: crack')
    logger.debug("create_data_yaml FINISH")
    return data_yaml_path

def retraining(model, dataset_path, out_path, name, epochs):
    """

    """
    # find data.yaml file. It should be just under the main dataset path.
    # It has to be prepared by the user for now but may be later i will create it automatically if it will be missing.
    logger.debug("retraining START")
    data_yaml_path = create_data_yaml(dataset_path)

    # augmentations are in args.yaml
    # implement resuming training from where it left
    ## first find the last weight
    # find if the name was already used
    """
    try:
        existing_names = sorted([n for n in os.listdir(out_path) if n.startswith(name)])
        logger.debug(f'existing_names: {existing_names}')
        last_name = existing_names[-1]
        logger.debug(f'last_name: {last_name}')
        last_weight_path = os.path.join(out_path, last_name, "weights", "last.pt")
    except:
        last_weight_path = os.path.join(out_path, name, "weights", "last.pt")
    """
    existing_names = [n for n in os.listdir(out_path) if n.startswith(name) and os.path.isdir(os.path.join(out_path, n))]

    if existing_names:
        #last_name = existing_names[-1]
        # get the last training by the creation time
        last_name = max(existing_names, key=lambda n: os.path.getmtime(os.path.join(out_path, n)))
        last_weight_path = os.path.join(out_path, last_name, "weights", "last.pt")
    else:
        last_weight_path = os.path.join(out_path, name, "weights", "last.pt")

    if os.path.isfile(last_weight_path):
        #load the last model from path_out location
        model = YOLO(last_weight_path)
        # check if training was finished
        finished_epochs = len(model.ckpt['train_results']['epoch'])
        logger.debug("The last weight number of epochs: %s", finished_epochs)
        logger.debug("Specified number of epochs to train the model: %s", epochs)
        if finished_epochs >= epochs:
            log_and_print(
                f"Training with this --run_ID: {name} is finished. Please change the --run_ID if you wish start a new training with a new name",
                logger, "info")
            return

        try:
            log_and_print(f"Resuming training from {last_weight_path}", logger, "info")
            model.train(resume=True)
        except Exception:
            logger.exception("Attempt to resume training failed.")
            return

    else:
        # Train from
        logger.debug("Starting training from submitted weight. No previous training was detected.")
        logger.info("Training started: epochs = %s", epochs)

        model.train(data=data_yaml_path, epochs=epochs, batch=-1, imgsz=1024, project=out_path, name=name,
                    lr0=0.01, lrf=0.01, seed=2, patience=500, overlap_mask=False, mask_ratio=2,
                    degrees=90, hsv_h=0.015, hsv_s=1.0, hsv_v=0.5, translate=0.1, scale=0.5, shear=20.0,
                    flipud=0.5, fliplr=0.5, copy_paste=0.2) # could be rewritten to pass training_args = {} and model.train(**train_args)
    logger.debug("retraining FINISH")

#########################################################
# FUNCTIONS evaluate trained weights
#########################################################
def save_res_yolo(res_yolo, out_file):
    # Save YOLO AP values for box and mask
    """
    box = res_yolo.box.all_ap
    mask = res_yolo.seg.all_ap
    out_data = np.array([box, mask]) # resulting array is 2,2,10 with out[0] being the box data
    np.save(out_file, out_data)
    """
    np.save(out_file, np.stack((res_yolo.box.all_ap, res_yolo.seg.all_ap)))

def load_annot_Polygons(yolo_annot_file, im_size, n_classes, cropUpandDown):
    logger.debug("load_annot_Polygons START")
    to_crop = int(im_size[0] * cropUpandDown)
    crop_box = shapely.geometry.box(0, to_crop, im_size[1], im_size[0] - to_crop)  # (minx, miny, maxx, maxy)

    logger.debug("crop_box bounds: %s", crop_box.bounds)
    contours, labels = load_annot(yolo_annot_file, im_size)
    polys = [[] for _ in range(n_classes)]
    for an, c_id in zip(contours, labels):
        an_poly_raw = shapely.geometry.Polygon(an)
        logger.debug("an_poly_raw area: %f", an_poly_raw.area)
        if not an_poly_raw.is_valid:
            an_poly_raw = shapely.make_valid(an_poly_raw)
            logger.debug("an_poly_raw area after .make_valid %f", an_poly_raw.area)
        try:
            an_poly = an_poly_raw.intersection(crop_box)
        except Exception as e:
            log_and_print(f"Polygon not valid after cropping with exception {e}", logger, "warning")
            continue
        poly_area = an_poly.area
        logger.debug("an_poly area %s", poly_area)
        if poly_area > 0:
            polys[int(c_id)].append(an_poly)

    #print("at the end", polys)
    #print("first ring bounds", polys[0][0].bounds)
    logger.debug("load_annot_Polygons FINISH")
    return polys

def get_detection_polys(image, model, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap):
    logger.debug("get_detection_polys START")
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
    clean_contours_cracks = clean_up_mask(detected_mask_cracks,
                                         min_mask_overlap=min_mask_overlap, is_ring=False)
    logger.debug("get_detection_polys FINISH")
    return (clean_contours_rings, clean_contours_cracks)

def _get_metrics(poly_d, poly_t, IoU_thresholds):
    # Calculate metrics per image per class
    # poly_d and poly_t are detected and truth shapely polygons
    # Precision as correctly detected/all detected
    # Recall as correctly detected/all real (ground truth) rings
    logger.debug("_get_metrics START")
    logger.debug(f"poly_d length {len(poly_d)}")
    logger.debug(f"poly_t length {len(poly_t)}")
    # ADD COMPREHENSION TO filter ONLY VALID POLYGONS
    #poly_t_v = [pT for pT in poly_t if shapely.is_valid(pT)] # just to see but remove, does not make sense they should be good
    #poly_d_v = [pD for pD in poly_d if shapely.is_valid_reason(pD)]
    #print(f'poly_t: {len(poly_t)}')

    if len(poly_t) == 0:
        NANs = np.repeat(np.nan, len(IoU_thresholds))
        P, R, IoU = NANs, NANs, NANs
    elif len(poly_d) == 0 and len(poly_t) != 0:
        zeros = np.repeat(0, len(IoU_thresholds))
        P, R, IoU = zeros, zeros, np.repeat(np.nan, len(IoU_thresholds))
    else:
        """"
        # this was before I managed to make it a oneliner
        #IoU_list_debug = []
        for pT in poly_t:
            #IoUs_temp_debug = []
            for pD in poly_d:
                logger.debug(f"intersection {pT.intersection(pD).area}")
                logger.debug(f"union {pT.union(pD).area}")
                logger.debug(f"pD area {pD.area}")
                logger.debug(f"pT area {pT.area}")
                IoU = pT.intersection(pD).area / pT.union(pD).area
                logger.debug(f"IoU {IoU}")
                if pD.area == 0:
                    crush
        
        # this was my oneliner for it
        IoU_list = [max((pT.intersection(pD).area / pT.union(pD).area for pD in poly_d)) for pT in poly_t]
        """
        # Trying supossedly more efficient version
        IoU_list = []
        available_detections = list(poly_d)
        for pT in poly_t:

            best_iou, best_idx = 0.0, None
            for idx, pD in enumerate(available_detections):

                inter = pT.intersection(pD).area
                if inter == 0:
                    continue

                iou = inter / (pT.area + pD.area - inter)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            IoU_list.append(best_iou)
            # remove matched detection
            if best_idx is not None:
                available_detections.pop(best_idx)

        IoU_list = np.array(IoU_list)
        logger.debug("IoU_list: %s", IoU_list)

        TPs = np.array([np.count_nonzero(IoU_list > threshold) for threshold in IoU_thresholds])
        logger.debug("TPs %s", TPs)
        P = TPs / len(poly_d)
        R = TPs / len(poly_t)
        IoU = [np.mean(IoU_list)] + np.repeat(np.nan, len(IoU_thresholds)-1).tolist() # make them same dimension to convert everything in np.array
    logger.debug("P: %s", P)
    logger.debug("R: %s", R)
    logger.debug("IoU: %s", IoU)
    logger.debug("_get_metrics FINISH")
    return P, R, IoU

def eval_image(image, model, yolo_annot_file, im_size, n_classes, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds):
    logger.debug("eval_image START")
    # get ground truth as polygons
    # GT and D are polygons by category. Ring in position 0 and crack in 1
    polys_gt = load_annot_Polygons(yolo_annot_file, im_size, n_classes, cropUpandDown)
    #print("polys_gt", polys_gt)
    # run detection and prepare the polygons
    polys_d = get_detection_polys(image, model, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap)
    if len(polys_gt) != n_classes:
        logger.warning("Expected %d classes but got %d",n_classes, len(polys_gt))
    if len(polys_gt) != len(polys_d):
        logger.error(f"Number of classes differs: GT={len(polys_gt)}, D={len(polys_d)}")
        raise ValueError(f"Number of classes differs: GT={len(polys_gt)}, D={len(polys_d)}")

    # in that order are also their metrics
    P, R, IoU = [], [], []
    for p_gt, p_d in zip(polys_gt, polys_d): # looping through polygons of rings and cracks
        Pt, Rt, IoUt = _get_metrics(poly_d=p_d, poly_t=p_gt, IoU_thresholds=IoU_thresholds)
        #print("len Pt, Rt, IoUt", len(Pt), len(Rt), len(IoUt))
        P.append(Pt)
        R.append(Rt)
        IoU.append(IoUt)

    logger.debug("eval_image FINISH")
    return (P, R, IoU)

def eval_dataset(data, model, n_classes, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds):
    logger.debug("eval_dataset START")
    supported_extensions = ('.tif', '.tiff', '.png')
    im_gen = (i for i in os.listdir(data) if os.path.splitext(i)[1] in supported_extensions and not i.startswith("."))
    results = []
    # im_name = "33627_201908231505-01(4)_8037a.tif"
    # im_name = "2019102817-01(12)_00015058a33_m01.tif" # many empty polygons loaded
    # im_name = "20115_00041007a_0_pSX1.965424714300121_pSY1.9655438706947042.tif"
    for im_name in im_gen:
        ## load image to extract the im size and other values
        log_and_print(f"Evaluating image  {im_name}", logger, "info")
        im_path = os.path.join(data, im_name)
        im = cv2.imread(im_path)
        if im is None:
            logger.warning("Failed to load image %s", im_path)
            continue
        im_size = im.shape[:2]
        yolo_annot_file = os.path.join(data, os.path.splitext(im_name)[0] + ".txt")

        ## load annotations by image name as shapely polygon per class
        results.append(eval_image(im, model, yolo_annot_file, im_size, n_classes, detection_rows,
                                  sliding_window_overlap, cropUpandDown, min_mask_overlap, IoU_thresholds))
    if not results:
        logger.error(f"No valid images found in {data}")
        raise ValueError(f"No valid images found in {data}")

    logger.debug("Processed %d images", len(results))
    out_array = np.nanmean(np.array(results), axis=0)  # average along the images
    logger.debug("results.shape: %s", out_array.shape)
    logger.debug("eval_dataset FINISH")
    return out_array
"""
def plot_results(res, IoU_thresholds, out_file_plot):
    n_classes = res.shape[1]
    linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
    for i in range(n_classes):
        precision = res[0][i]
        recall = res[1][i]
        plt.plot(IoU_thresholds, precision, ls=linestyle[i], c='b')
        plt.plot(IoU_thresholds, recall, ls=linestyle[i], c='orange')
    plt.xlabel('IoU threshold')
    plt.legend(['Precision', 'Recall'])
    plt.grid()
    #plt.show()
    plt.savefig(out_file_plot)
    plt.close()
"""
def plot_results(res, IoU_thresholds, out_file_plot):
    logger.debug("plot_results START")
    linestyle = ['solid', 'dashed'] #, 'dashdot', 'dotted']
    class_names = ["Rings", "Cracks"]
    # specify basics
    f, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlabel('IoU threshold')

    dummy_lines_for_legend = [] # needed only for linestyle legend

    for i in range(res.shape[1]):
        #precision = res[0][i]
        #recall = res[1][i]
        precision, recall = res[:2, i]
        ax.plot(IoU_thresholds, precision, ls=linestyle[i], c='tab:blue')
        ax.plot(IoU_thresholds, recall, ls=linestyle[i], c='tab:orange')
        dummy_lines_for_legend.append(ax.plot([], [], c="black", ls=linestyle[i])[0]) # needed only for linestyle legend

    # add legend for colour and linestyle
    legend1 = ax.legend(['Precision', 'Recall'], loc=(0.76, 0.85))
    #ax.legend([dummy_lines_for_legend[i] for i in [0, 1]], class_names, loc=(0.76, 0.7))
    ax.legend(dummy_lines_for_legend[:2], class_names, loc=(0.76, 0.7))
    ax.add_artist(legend1)

    # add grid to the plot background
    ax.grid()
    #plt.show()
    f.savefig(out_file_plot)
    plt.close(f)
    logger.debug("plot_results FINISH")

def evaluate_weight(data, weight_path, res_out_path, n_classes, detection_rows,
                    sliding_window_overlap, cropUpandDown, min_mask_overlap):
    weight_name = os.path.basename(weight_path)
    log_and_print(f"Evaluating weight  {weight_name}", logger, "info")
    # check if results for this weight already exist
    csv_file_out = os.path.join(res_out_path, weight_name.replace(".pt", ".csv"))
    out_file_plot = os.path.join(res_out_path, weight_name.replace(".pt", ".png"))
    if os.path.isfile(csv_file_out) and os.path.isfile(out_file_plot):
        log_and_print(f"Results for weight: {weight_name} already exist", logger, "info")
        return

    # prepare model
    model = YOLO(weight_path)

    IoU_thresholds = np.arange(0.5, 1, 0.05)

    res_arr = eval_dataset(data, model, n_classes, detection_rows, sliding_window_overlap,
                                               cropUpandDown, min_mask_overlap, IoU_thresholds)

    # res_arr shape is 3, 2, 10. In first dimension is Precision, Recal, IoU in that order, second dimension ring, crack
    df_out = pd.DataFrame({
        "IoU_threshold": IoU_thresholds,
        "Ring_Precision": res_arr[0, 0],
        "Ring_Recall": res_arr[1, 0],
        "Ring_IoU": res_arr[2, 0],
        "Crack_Precision": res_arr[0, 1],
        "Crack_Recall": res_arr[1, 1],
        "Crack_IoU": res_arr[2, 1],
    })

    # summary_out = np.nanmean(res_arr, axis=2) # averages over IoU thresholds
    # summary_out.tofile(csv_file_out, sep=',')  # , format='%10.5f')
    df_out.to_csv(csv_file_out, index=False, float_format="%.4f")
    plot_results(res_arr, IoU_thresholds, out_file_plot)

def evaluate_training(dataset_path, out_path, name, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap):
    log_and_print("Started evaluation of the new weights", logger, "info")
    # prepare input variables
    data = os.path.join(dataset_path, "val")
    n_classes = 2
    # it will check all generated weights and make one subfolder with figure and table per each
    weights_path = os.path.join(out_path, name, "weights")
    #weights_list = os.listdir(weights_path)
    weights_list = (f for f in os.listdir(weights_path) if f.endswith(".pt"))
    res_out_path = os.path.join(out_path, name, "eval")
    # make if does not exist
    os.makedirs(res_out_path, exist_ok=True)

    for weight_name in weights_list:
        weight_path = os.path.join(weights_path, weight_name)
        try:
            evaluate_weight(data, weight_path, res_out_path, n_classes, detection_rows,
                            sliding_window_overlap, cropUpandDown, min_mask_overlap)
        except Exception:
            logger.exception("Evaluation failed for %s", weight_name)
            continue

    logger.debug("evaluate_training FINISH")


