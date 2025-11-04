"""
Prepares annotations from .xml files from CVAT_annotations folder and distributes them based on the image names in
train and val folders.
In the CVAT annotation file the ring boundary has to be a polyline RingBndy and crack as polygone labeled CrackPoly.
"""

############# IMPORTS #################
import os
from lxml import etree
import shapely
from shapely.geometry import LineString, Polygon, box
from ultralytics import YOLO
import cv2
import sys
import matplotlib.pyplot as plt
import logging
logging.getLogger('PIL').setLevel(logging.WARNING)
#from datetime import datetime
import numpy as np
logger = logging.getLogger(__name__)

# Import custom functions
ROOT_DIR = os.path.abspath("../")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR) # To find local version of the library

from functions.processing_functions import (apply_mask, convert_to_binary_mask, load_annot,
                                            sliding_window_detection_multirow, clean_up_mask)


#########################################################
# FUNCTIONS CVAT annotations
#########################################################
def collect_annotations(CVAT_folder):
    # extracts individual image annotations dictionaries from all xml files and put them in one list
    logger.debug("collect_annotations START")
    all_image_xml_list = []
    CVAT_annot_file_list = [f for f in os.listdir(CVAT_folder) if not f.startswith('.')]  # ignore hidden files
    # make sure they are xml files
    for xml_name in CVAT_annot_file_list:
        if xml_name.endswith('.xml'):
            # load xml file
            print(f"Loading annotation file: {xml_name}")
            logger.info(f"Loading annotation file: {xml_name}")
            with open(os.path.join(CVAT_folder, xml_name)) as file:
                root = etree.parse(file).getroot()

            for i in root.iter('image'):
                logger.debug(f'Image name: {[value for key, value in i.items() if key == "name"]}')
                all_image_xml_list.append(i)
        else:
            print(f"{xml_name} not proccesed because it`s not valid annotation file")
            logger.info(f"{xml_name} not proccesed because it`s not valid annotation file")

    logger.debug("collect_annotations FINISH")
    return all_image_xml_list

def polylinetopolygon(polyline_str, width, height, buffer=10):
    # takes polyline string form CVAT xml.
    # Using shapely package it transforms line into polygon
    # Output x and y coords of polygone.
    # width and height of the image ensure that polygone is not exceeding image
    # buffer=0 will export only the lines coordinates for rings
    logger.debug("polylinetopolygon START")
    points = polyline_str.split(";")
    #coords = list((x, y) for point.split(",") in points)
    xy_coords = list(tuple(map(float, (point.split(",")))) for point in points)
    polyline = LineString(xy_coords)
    im_box = box(1, 1, int(width)-1, int(height)-1) # to be sure I crop it one pixel inside the image
    if buffer>0:
        polygon_ring = polyline.buffer(buffer)
        polygon_clean = polygon_ring.intersection(im_box)
        x, y = polygon_clean.exterior.coords.xy
    else:
        x, y = polyline.coords.xy
    logger.debug("polylinetopolygon FINISH")
    return list(x), list(y)

def preparepolygon(polygon_str):
    # Output x and y coords of polygone.
    logger.debug("preparepolygon START")
    points = polygon_str.split(";")
    xy_coords = list(tuple(map(float, (point.split(",")))) for point in points)
    polygon = Polygon(xy_coords)
    x, y = polygon.exterior.coords.xy
    logger.debug("preparepolygon FINISH")
    return list(x), list(y)

def prepare_annotations(dataset_path, annot_list, buffer=10, overwrite_existing=False):
    logger.debug("prepare_annotations START")
    dir_list = os.listdir(dataset_path)
    supported_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    im_list = [f for f in dir_list if f.endswith(supported_extensions) and not f.startswith('.')]

    for i in annot_list:
        # get image name to control the loop
        [image_name] = [value for key, value in i.items() if key == "name"]
        annot_txt_file = os.path.join(dataset_path, os.path.splitext(image_name)[0] + ".txt")
        if image_name not in im_list:
            continue
        elif os.path.isfile(annot_txt_file) and overwrite_existing==False:
            continue
        else:
            print(f"Processing annotation file of image: {image_name}")
            logger.info(f"Processing annotation file of image: {image_name}")
            # create a text file named filename.txt
            f = open(annot_txt_file, "w+")
            for key, value in i.items():
                if key == "width":
                    im_width = value
                if key == "height":
                    im_height = value

            # get polylines which are Ring annotations
            for poly_tag in i.iter('polyline'):
                logger.debug(f"poly_tag: {poly_tag}")
                for key, value in poly_tag.items():
                    # print(key)
                    # print(value)
                    if key == "label":
                        # print(value)
                        if value != "RingBndy":
                            print("Warning: Label is not RingBndy. Continue assuming all polylines are rings")
                            logger.warning("Warning: Label is not RingBndy. Continue assuming all polylines are rings")
                    if key == "points":
                        all_points_x, all_points_y = polylinetopolygon(polyline_str=value, width=im_width, height=im_height,
                                                                       buffer=buffer)
                        # normalize the coordinates by image size
                        all_points_x_norm = [x / int(im_width) for x in all_points_x]
                        all_points_y_norm = [y / int(im_height) for y in all_points_y]
                        # create and save the line of this ring
                        line = "\n0 " # Ring is 0 crack is 1
                        for point_index in range(len(all_points_x_norm)):
                            line += str(all_points_x_norm[point_index]) + " "
                            line += str(all_points_y_norm[point_index]) + " "
                        f.write(line)

            # here the preparation of polygons for cracks
            for poly_tag in i.iter('polygon'):
                logger.debug(f"poly_tag: {poly_tag}")
                for key, value in poly_tag.items():
                    # print(key)
                    # print(value)
                    if key == "label":
                        # print(value)
                        if value != "CrackPoly":
                            print("Warning: Label is not CrackPoly. Continue assuming all polygon are crack")
                            logger.warning("Warning: Label is not CrackPoly. Continue assuming all polygon are crack")
                    if key == "points":
                        all_points_x, all_points_y = preparepolygon(polygon_str=value)
                        # normalize the coordinates by image size
                        all_points_x_norm = [x / int(im_width) for x in all_points_x]
                        all_points_y_norm = [y / int(im_height) for y in all_points_y]
                        # create and save the line of this ring
                        line = "\n1 "  # Ring is 0 crack is 1
                        for point_index in range(len(all_points_x_norm)):
                            line += str(all_points_x_norm[point_index]) + " "
                            line += str(all_points_y_norm[point_index]) + " "
                        f.write(line)
            f.close()
    logger.debug("prepare_annotations FINISH")
######## MAIN ############################
def prepare_all_annotations(dataset_path, buffer=10, overwrite_existing=True):
    logger.debug("prepare_all_annotations START")
    CVAT_ANNOT_PATH = os.path.join(dataset_path, 'CVAT_annotations')
    all_annot_list = collect_annotations(CVAT_folder=CVAT_ANNOT_PATH)

    subset_list = ['train', 'val']
    for subset in subset_list:
        FOLDER_PATH = os.path.join(dataset_path, subset)
        prepare_annotations(dataset_path=FOLDER_PATH, annot_list=all_annot_list, buffer=buffer, overwrite_existing=overwrite_existing)
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
    logger.debug(f'path: {data_yaml_path}')
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
    try:
        existing_names = sorted([n for n in os.listdir(out_path) if n.startswith(name)])
        logger.debug(f'existing_names: {existing_names}')
        last_name = existing_names[-1]
        logger.debug(f'last_name: {last_name}')
        last_weigth_path = os.path.join(out_path, last_name, "weights", "last.pt")
    except:
        last_weigth_path = os.path.join(out_path, name, "weights", "last.pt")

    if os.path.isfile(last_weigth_path):
        #load the last model from path_out location
        model = YOLO(last_weigth_path)
        logger.debug(f"the last weight number of epochs: {len(model.ckpt['train_results']['epoch'])}")
        logger.debug(f"specified number of epochs to train the model: {epochs}")
        try:
            model.train(resume=True)
            #model.train(data=data_yaml_path, epochs=epochs, imgsz=640, project=out_path, name=name, resume=True)
        except Exception as e:
            print("Training with this --run_ID is finished. Please change the --run_ID if you wish start a new training with a new name")
            logger.info("Training with this --run_ID is finihsed. Please change the --run_ID if you wish start a new training with a new name")
            logger.error(e)

    else:
        # Train from
        logger.debug(f"Starting training from submitted weight. No previous training was detected.")
        model.train(data=data_yaml_path, epochs=epochs, batch=-1, imgsz=640, project=out_path, name=name,
                    lr0=0.01, lrf=0.01, seed=2, patience=500, overlap_mask=False, mask_ratio=2,
                    degrees=90, hsv_h=0.015, hsv_s=1.0, hsv_v=0.5, translate=0.1, scale=0.5, shear=20.0,
                    flipud=0.5, fliplr=0.5, copy_paste=0.2)
    logger.debug("retraining FINISH")

#########################################################
# FUNCTIONS evaluate trained weights
#########################################################
def save_res_yolo(res_yolo, out_file):
    box = res_yolo.box.all_ap
    mask = res_yolo.seg.all_ap
    out_data = np.array([box, mask]) # resulting array is 2,2,10 with out[0] being the box data
    np.save(out_file, out_data)

def load_annot_Polygons(yolo_annot_file, im_size, n_classes, cropUpandDown):
    annot = load_annot(yolo_annot_file, im_size)
    if cropUpandDown > 0:
        to_crop = int(im_size[0] * cropUpandDown)
        crop_box = shapely.geometry.box(0, to_crop, im_size[1], im_size[0] - to_crop)  # (minx, miny, maxx, maxy)
    else:
        crop_box = shapely.geometry.box(0, 0, im_size[1], im_size[0])  # (minx, miny, maxx, maxy)

    logger.debug(f"crop_box bounds{crop_box.bounds}")
    polys = [[] for _ in range(n_classes)]
    for an, c_id in zip(annot[0], annot[1]):
        an_poly_raw = shapely.geometry.Polygon(an)
        logger.debug(f"an_poly_raw area {an_poly_raw.area}")
        try:
            an_poly = an_poly_raw.intersection(crop_box)
        except Exception as e:
            logger.warning(f'Polygon not valid after cropping with exception {e}')
            print(f'Polygon not valid after cropping with exception {e}')
            continue
        poly_area = an_poly.area
        logger.debug(f"an_poly area {poly_area}")
        if poly_area > 0:
            polys[int(c_id)].append(an_poly)

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
        IoU_list_debug = []
        for pT in poly_t:
            IoUs_temp_debug = []
            for pD in poly_d:
                logger.debug(f"intersection {pT.intersection(pD).area}")
                logger.debug(f"union {pT.union(pD).area}")
                logger.debug(f"pD area {pD.area}")
                logger.debug(f"pT area {pT.area}")
                IoU = pT.intersection(pD).area / pT.union(pD).area
                logger.debug(f"IoU {IoU}")
                if pD.area == 0:
                    crush

        IoU_list = [max((pT.intersection(pD).area / pT.union(pD).area for pD in poly_d))
                         for pT in poly_t]

        TPs = np.array([len(np.where(IoU_list > IoU_threshold)[0]) for IoU_threshold in IoU_thresholds])
        logger.debug(f"TPs {TPs}")
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
        logger.info(f"evaluating image  {im_name}")
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

def evaluate_training(dataset_path, out_path, name, detection_rows, sliding_window_overlap, cropUpandDown, min_mask_overlap):
    # prepare input variables
    data = os.path.join(dataset_path, "val")
    n_classes = 2
    IoU_thresholds = np.arange(0.5, 1, 0.05)
    # it will check all generated weights and make one subfolder with figure and table per each
    weights_path = os.path.join(out_path, name, "weights")
    weights_list = os.listdir(weights_path)
    res_out_path = os.path.join(out_path, name, "eval")
    # make if does not exist
    if not os.path.isdir(res_out_path):
        os.makedirs(res_out_path)

    for weight_name in weights_list:
        # prepare model
        weight_path = os.path.join(weights_path, weight_name)
        model = YOLO(weight_path)
        res_arr = eval_dataset(data, model, n_classes, detection_rows, sliding_window_overlap, cropUpandDown,
                               min_mask_overlap, IoU_thresholds)

        summary_out = np.nanmean(res_arr, axis=2)
        csv_file_out = os.path.join(res_out_path, weight_name.replace(".pt", ".csv"))
        summary_out.tofile(csv_file_out, sep=',')  # , format='%10.5f')
        out_file_plot = os.path.join(res_out_path, weight_name.replace(".pt", ".png"))
        plot_results(res_arr, IoU_thresholds, out_file_plot)



