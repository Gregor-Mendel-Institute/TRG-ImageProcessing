"""
Load tiff image of whole core.
Run detections for squares in a form of sliding window with certain overlap to prevent problems of having ring directly at the edge.
Run detection separately for the best model for Ring detection and the best for Crack.
Fuse all detections in one mask layer and consequently attach all of them to each other creating mask layer
of the size of original image. The detection confidance needs to be set in the config!
Print the image with mask over it.

FOR TESTING Mac
conda activate TreeRingCNNtest &&
cd /Users/miroslav.polacek/github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing &&
python3 postprocessingCracks.py --dpi=12926 --run_ID=RUN_ID_SOME_VALUE --input=/Users/miroslav.polacek/Pictures/whole_core_examples --weight=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/mask_rcnn_treeringcrackscomb_0222.h5 --output_folder=/Users/miroslav.polacek/Documents/CNNTestRuns

FOR TESTING MANJARO
conda activate TreeRingCNN &&
cd /home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing &&
python3 postprocessingCracksRings.py --dpi=12926 --run_ID=RUN_ID_SOME_VALUE --input=/home/miroslavp/Pictures/whole_core_examples --weightRing=/home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 --weightCrack=/home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 --output_folder=/home/miroslavp/Documents/CNNTestRuns

"""

#######################################################################
#Arguments
#######################################################################
import argparse

    # Parse command line arguments
parser = argparse.ArgumentParser(
        description='Segmentation of whole core')

parser.add_argument('--dpi', required=True,
                    help="DPI value for the image")

parser.add_argument('--run_ID', required=True,
                    help="Run ID")

parser.add_argument('--input', required=True,
                    metavar="/path/to/image/",
                    help="Path to image file of folder")

parser.add_argument('--weightRing', required=True,
                    metavar="/path/to/weight/file",
                    help="Path to weight file")

parser.add_argument('--weightCrack', required=True,
                    metavar="/path/to/weight/file",
                    help="Path to weight file")

parser.add_argument('--output_folder', required=True,
                    metavar="/path/to/out/folder",
                    help="Path to output folder")

args = parser.parse_args()

#######################################################################
# Imports and prepare model
#######################################################################
import os
import sys
import random
import math
import re
import cv2
import json
import time
import skimage
import pandas as pd
import numpy as np
#import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shapely
from shapely.geometry import box
from shapely.ops import nearest_points
import scipy
from scipy import optimize
from datetime import datetime
from src_get_centerline import get_centerline
from operator import itemgetter

# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log

from DetectionConfig import TreeRing_onlyRing
from DetectionConfig import TreeRing_onlyCracks

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

configRing = TreeRing_onlyRing.TreeRingConfig()
configCrack = TreeRing_onlyCracks.TreeRingConfig()

class InferenceConfig(configRing.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

configRing = InferenceConfig()
configRing.display()

class InferenceConfig(configCrack.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

configCrack = InferenceConfig()
configCrack.display()
# Create model in inference mode
modelRing = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configCrack)
modelCrack = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configRing)
# Load weights
weights_path_Ring = args.weightRing
weights_path_Crack = args.weightCrack

print("Loading weights ")
modelRing.load_weights(weights_path_Ring, by_name=True)
modelCrack.load_weights(weights_path_Crack, by_name=True)

#define class names
class_names = ['BG', 'ring']

#######################################################################
# apply mask to an original image
#######################################################################
def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image.
    """
    color = (0.7, 0.0, 0.0)

    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    color = (0.0, 0.7, 0.0)
    for c in range(3):
        image[:, :, c] = np.where(mask == 2,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    color = (0.0, 0.0, 0.7)
    for c in range(3):
        image[:, :, c] = np.where(mask > 2,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
#######################################################################
# write run information
#######################################################################
# should find the log file and add data to it
def write_run_info(string):
    #get name of the log file in the output dir
    out_dir = os.path.join(args.output_folder, args.run_ID)
    run_ID = args.run_ID
    log_files = []
    for f in os.listdir(out_dir):
        if f.startswith("CNN_" + run_ID):
            path_f = os.path.join(out_dir, f)
            log_files.append(path_f)

    # sort files by creation times
    log_files.sort(key=os.path.getctime)
    #print("sorted log files list", log_files)

    log_file_name = log_files[-1]
    with open(log_file_name,"a") as f:
        print(string, file=f)

############################################################################################################
# Sliding window detection with rotation of each part of image by 90 and 45 degrees and combining the output
############################################################################################################
def sliding_window_detection(image, overlap = 0.5, cropUpandDown = 0):
    write_run_info("Sliding window overlap = {} and cropUpandDown = {}".format(overlap, cropUpandDown))
    #crop image top and bottom to avoid detectectig useles part of the image
    imgheight_origin, imgwidth_origin = image.shape[:2]

    #print('image shape', image.shape[:2])
    to_crop = int(imgheight_origin*cropUpandDown)
    new_image = image[to_crop:(imgheight_origin-to_crop), :, :]
    #print('new image shape', new_image.shape)

    imgheight_for_pad, imgwidth_for_pad = new_image.shape[:2]

    # add zero padding at the begining and the end according to overlap so every part of the picture is detected same number of times
    zero_padding_front = np.zeros(shape=(imgheight_for_pad, int(imgheight_for_pad-(imgheight_for_pad*overlap)),3))
    zero_padding_back = np.zeros(shape=(imgheight_for_pad, imgheight_for_pad,3))
    #print('padding', zero_padding.shape)
    im_padded = np.concatenate((zero_padding_front, new_image, zero_padding_back), axis=1)

    imgheight, imgwidth = im_padded.shape[:2]
    #print('im_after_pad', im_padded.shape)

    looping_range = range(0,imgwidth, int(imgheight-(imgheight*overlap)))
    looping_list = [i for i in looping_range if i < int(imgheight_for_pad-(imgheight_for_pad*overlap)) + imgwidth_origin]
    #print('looping_list', looping_list)

    combined_masks_per_class = np.empty(shape=(imgheight, imgwidth,0))
    for model in [modelRing, modelCrack]:
        the_mask = np.zeros(shape=(imgheight, imgwidth)) #combine all the partial masks in the final size of full tiff
        #print('the_mask', the_mask.shape)
        for i in looping_list: #defines the slide value

            #print("i", i)
            # crop the image
            cropped_part = im_padded[:imgheight, i:(i+imgheight)]
            #print('cropped_part, i, i+imheight', cropped_part.shape, i, i+imgheight)

            # run detection on the cropped part of the image
            results = model.detect([cropped_part], verbose=0)
            r = results[0]
            r_mask = r['masks']
            #visualize.display_instances(cropped_part, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']) #just to check

            # rotate image 90 and run detection
            cropped_part_90 = skimage.transform.rotate(cropped_part, 90, preserve_range=True).astype(np.uint8)
            results1 = model.detect([cropped_part_90], verbose=0)
            r1 = results1[0]
            r1_mask = r1['masks']
            #visualize.display_instances(cropped_part_90, r1['rois'], r1['masks'], r1['class_ids'], class_names, r1['scores']) #just to check

            # rotate image 45 and run detection
            cropped_part_45 = skimage.transform.rotate(cropped_part, angle = 45, resize=True).astype(np.uint8)
            results2 = model.detect([cropped_part_45], verbose=0)
            r2 = results2[0]
            r2_mask = r2['masks']

            ## flatten all in one layer. This should be one layer of zeroes and ones. If applying som e cleaning i would do it before this point or cleaning of the final one might be even better
            maskr = np.zeros(shape=(imgheight, imgheight))
            #print('maskr_empty', maskr)
            nmasks = r_mask.shape[2]
            #print(nmasks)
            for m in range(0,nmasks):
                maskr = maskr + r_mask[:,:,m]
                #print(maskr.sum())

            maskr1 = np.zeros(shape=(imgheight, imgheight))
            nmasks1 = r1_mask.shape[2]
            for m in range(0,nmasks1):
                maskr1 = maskr1 + r1_mask[:,:,m]
            ## rotate maskr1 masks back
            maskr1_back = np.rot90(maskr1, k=-1)
            # beware different dimensions!!!

            imheight2 = r2_mask.shape[0]
            nmasks2 = r2_mask.shape[2]
            maskr2 = np.zeros(shape=(imheight2, imheight2))
            for m in range(0,nmasks2):
                maskr2 = maskr2 + r2_mask[:,:,m]
            #rotate back
            maskr2_back = skimage.transform.rotate(maskr2, angle = -45, resize=False)
            #crop to the right size
            to_crop = int((imheight2 - imgheight)/2)

            maskr2_back_cropped = maskr2_back[to_crop:(to_crop+int(imgheight)), to_crop:(to_crop+int(imgheight))]

            ## put both togather. before this can one turn everything to 0 and 1 or leave it so far but may be it is not too useful to detect very overlapping areas
            combined_mask = maskr1_back + maskr + maskr2_back_cropped
            # merge with the relevant position of the big mask and overlap with previous one
            cropped_the_mask = the_mask[:imgheight, i:(i+imgheight)]
            #print('the_mask_piece', cropped_the_mask.shape)
            all_masks_combined = combined_mask + cropped_the_mask
            #print("all_masks_combined", all_masks_combined.shape)
            #all_masks_combined = np.reshape(all_masks_combined, (all_masks_combined.shape[0],all_masks_combined.shape[1],1))
            #print('middle', all_masks_combined.shape)
            end_the_mask = the_mask[:imgheight, (i+imgheight):]

            if i == 0: # to solve the begining
                the_mask = np.concatenate((all_masks_combined, end_the_mask), axis=1)
            else:
                begining_the_mask = the_mask[:imgheight, :i]
                the_mask = np.concatenate((begining_the_mask, all_masks_combined, end_the_mask), axis=1)
            #print("the_mask.shape", the_mask.shape)

        the_mask = np.reshape(the_mask, (the_mask.shape[0],the_mask.shape[1],1))
        combined_masks_per_class = np.append(combined_masks_per_class, the_mask, axis=2)
        #print("combined_masks_per_class.shape",combined_masks_per_class.shape)

    # First remove the padding
    pad_front = zero_padding_front.shape[1]
    #print('front', pad_front)
    pad_back = zero_padding_back.shape[1]
    the_mask_clean = combined_masks_per_class[:,pad_front:-pad_back,:]
    #print('the_mask_clean.shape', the_mask_clean.shape)
    #print('the_mask_clean', the_mask_clean.shape)

    #here you have to concatanete the top and buttom to fit the original image

    missing_part = int((imgheight_origin - the_mask_clean.shape[0])/2)
    to_concatenate = np.zeros(shape=(missing_part, imgwidth_origin,2))
    #print("to_concatenate", to_concatenate.shape)
    the_mask_clean_origin_size = np.concatenate((to_concatenate, the_mask_clean, to_concatenate),axis=0)
    #print('the_mask_clean_origin_size', the_mask_clean_origin_size.shape)
    #plt.imshow(the_mask_clean) # uncomment to print mask layer
    #plt.show()
    # TO PRINT THE MASK OVERLAYING THE IMAGE

    return the_mask_clean_origin_size[:,:,0], the_mask_clean_origin_size[:,:,1]

#######################################################################
# Extract distances from the mask
#######################################################################
def clean_up_mask(mask, is_ring=True):
    # detects countours of the masks, removes small contours, fits circle to individual contours and estimates the pith, skeletonizes the detected contours

    # make the mask binary
    binary_mask = np.where(mask > 2, 255, 0) # this part can be cleaned to remove some missdetections setting condition for >=2
    print("binary_mask shape", binary_mask.shape)
    #plt.show()
    #type(binary_mask)
    uint8binary = binary_mask.astype(np.uint8).copy()

    # Older version of openCV has slightly different syntax i adjusted for it here
    if int(cv2.__version__.split(".")[0]) < 4:
        _, contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print('contour_shape:', len(contours))

    #### here i extract dimensions and angle of individual contours bigger than threshold
    imgheight, imgwidth = mask.shape[:2]
    if is_ring==True:
        min_size_threshold = imgheight/5 #will take only contours that are bigger than 1/5 of the image
    else:
        min_size_threshold = 1
    contours_filtered = []
    x_mins = []
    for i in range(0, len(contours)):
        x_only = []
        for p in range(len(contours[i])):
            [[x,y]] = contours[i][p]
            x_only.append(x)
        x_min = np.min(x_only)
        #remove those that are too short
        rect = cv2.minAreaRect(contours[i])
        #print(rect)
        imgheight, imgwidth = mask.shape[:2]
        #print(imgheight)
        dim1, dim2 = rect[1]
        dim_max = max([dim1, dim2])
        dim_sum = dim1 + dim2 # should capture size and curvature better then just a length
        if dim_max > min_size_threshold:
            contours_filtered.append(contours[i])
            x_mins.append(x_min)

    print('Filtered_contours:', len(contours_filtered))

    #print(contours_filtered[0])

    #### Extract longest contour to use for center estimate
    if is_ring==True:
        contourszip = zip(x_mins, contours_filtered)
        contours_out = [x for _,x in sorted(contourszip, reverse = False)]
    else:
        contours_out = contours_filtered
        
    return contours_out # Returns filtered and orderedt contours

#######################################################################
# Finds centerines in contours
#######################################################################
def find_centerlines(clean_contours):
    # find ceneterlines

    #first need to reorganise the data
    contours_tuples = []
    x_mins = []
    for i in range(len(clean_contours)):
        xy_tuples = []
        x_only = []
        for p in range(len(clean_contours[i])):
            [[x,y]] = clean_contours[i][p]
            xy_tuples.append((x,y))
            x_only.append(x)
        x_min = np.min(x_only)
        x_mins.append(x_min)
        contours_tuples.append(xy_tuples)

    # order contours by x_min
    contourszip = zip(x_mins, contours_tuples)
    contours_tuples = [x for _,x in sorted(contourszip, key=itemgetter(0))]

    centerlines = []
    for i in range(len(contours_tuples)): #range(len(contours_tuples)):
        #print('ring_contour:', i)
        contour = contours_tuples[i]
        #print('contour:', contour)
        polygon = shapely.geometry.Polygon(contour)
        #x0, y0 = polygon.exterior.coords.xy
        #plt.plot(x0, y0)
        #exterior_coords = polygon.exterior.coords
        #print('polygon_points:', len(exterior_coords))
        try:
            cline = get_centerline(polygon, segmentize_maxlen=10, max_points=300, simplification=0.05)
        except Exception as e:
            write_run_info('Centerline of the ring {} failed with exception {}'.format(i, e))
            print('Centerline of the ring {} failed with exception {}'.format(i, e))
            continue
        #xc,yc = cline.coords.xy
        #plt.plot(xc,yc,'g')
        #print('cline done')
        #to remove horizontal lines
        _, miny, _, maxy = cline.bounds
        line_y_diff = maxy - miny
        if line_y_diff < 100: #this threshold should be adjusted
            continue
        else:
            centerlines.append(cline)

    ## Cut off upper and lower part of detected lines. It should help with problems of horizontal ends of detections
    to_cut_off = 0.01 #based on examples that i used for testing
    Multi_centerlines_to_crop = shapely.geometry.MultiLineString(centerlines)
    minx, miny, maxx, maxy = Multi_centerlines_to_crop.bounds
    px_to_cut_off = (maxy-miny)*to_cut_off
    #print('minx, miny, maxx, maxy', minx, miny, maxx, maxy)
    frame_to_crop = shapely.geometry.box(minx, miny+px_to_cut_off, maxx, maxy-px_to_cut_off)
    Multi_centerlines = Multi_centerlines_to_crop.intersection(frame_to_crop)
    # to check if it cropps something
    #minx, miny, maxx, maxy = Multi_centerlines.bounds
    #print('minx, miny, maxx, maxy after', minx, miny, maxx, maxy)

    return Multi_centerlines

#######################################################################
# Turn contours into lines and find nearest points between them for measure
#######################################################################
# return table of distances or paired point coordinates?
def measure_contours(Multi_centerlines, image):
    # find nearest_points for each pair of lines
    imgheight, imgwidth = image.shape[:2]
    print('imgheight, imgwidth', imgheight, imgwidth)
    write_run_info("Image has height {} and width {}".format(imgheight, imgwidth))
    write_run_info("{} ring boundries were detected".format(len(Multi_centerlines)))

    ## Split samples that are crosing center into two then turn the other part around
    #may be I need image dimensions

    angle_index = [] #average angle of the section of centerlines.
    PlusMinus_index = []
    frame_width = imgheight * .75
    sliding = frame_width * .5 #how much is the frame sliding in every frame
    #print('frame_width', frame_width)
    number_of_segments = int(imgwidth/sliding)
    #print('number_of_segments', number_of_segments)
    #plt.imshow(image)
    for i in range(0, number_of_segments):
        #print('loop_number', i)
        #get the frame
        frame_poly = shapely.geometry.box(i*sliding, 0, (i*sliding)+frame_width, imgheight)
        cut_point = i*sliding+(frame_width*.5) #better to get cutting point here and use instead of frame number
        #print('cutting_point', cutting_point)
        #print('frame_exterior_xy',frame_poly.exterior.coords.xy)
        #x, y = frame_poly.exterior.coords.xy
        #plt.plot(x,y)
        #get lines inside of the frame
        intersection = Multi_centerlines.intersection(frame_poly)
        #print('intersection:', intersection.geom_type)
        if intersection.geom_type=='LineString':
            x, y = intersection.coords.xy

            #line_coords = sorted(line_coords, reverse = True)# i think i do not need this for slope
            #print('sorted:', line_coords)

            x_dif = abs(x[-1] - x[0])
            if x_dif < frame_width*.20: #This should be adjusted now it should skip this frame if a line is less then 20% of the frame width
                #print(i, 'th is too short')
                continue
            else:
                #print(i, "th frame is simple")
                slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x, y)
                #write_run_info("slope:{}".format(slope))
                angle_index.append([abs(slope), cut_point]) #I add also i here so i can identify distance later
                if slope > 0 and slope < 2:
                    PlusMinus = 1
                elif slope < 0 and slope > -2:
                    PlusMinus = 0
                else:
                    PlusMinus = []
                PlusMinus_index.append([PlusMinus, cut_point])

        elif intersection.geom_type=='MultiLineString':
            slopes = []
            for l in range(len(intersection.geoms)):
                x, y = intersection.geoms[l].coords.xy
                x_dif = abs(x[-1] - x[0])
                #print('loop number and xy coords:',i, l, x, y)
                if x_dif < frame_width*.20: #This should be adjusted now it should skip this frame is line is less then 20% of the frame width
                    #print(i, 'th is too short')
                    continue
                else:
                    #print(i, "th frame is complex")
                    slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(x, y)
                    #write_run_info("slope:{}".format(slope))

                    #print('linestring_x',x_int)
                    #plt.plot(x_int, y_int)
                slopes.append(slope)

            #print('slopes before mean', slopes)
            mean_slopes = np.mean(slopes)
            #print('mean_slopes', mean_slopes)
            if np.isnan(mean_slopes):
                continue
            else:
                angle_index.append([abs(mean_slopes), cut_point])
                if mean_slopes > 0 and slope < 2:
                    PlusMinus = 1
                elif mean_slopes < 0 and slope > -2:
                    PlusMinus = 0
                else:
                    PlusMinus = []
                PlusMinus_index.append([PlusMinus,cut_point])
        else:
            #print('second else????')
            continue

    #get cutting point as a global minimum of polynomial function fit to angle indexes
    #print('angle_index', angle_index)

    angle = 1 #remove this after testing
    #print('PlusMinus_index', PlusMinus_index)

    # find the middle by the change in a slope of the lines
    if angle >= 0.3:
        cutting_point = []
        test_seq1 = [0,0,1,1]
        test_seq2 = [1,1,0,0]
        PlusMinus = [x for x,_ in  PlusMinus_index]
        for i in range(len(PlusMinus_index)):
            pm_seq = PlusMinus[i:i+len(test_seq1)]
            if pm_seq != test_seq1 and pm_seq != test_seq2:
                continue
            if angle == 0:
                print('Several cutting points identified, needs to be investigated!')
                write_run_info('Several cutting points identified, needs to be investigated!')
                break
            cutting_point = PlusMinus_index[i+1][1] + ((PlusMinus_index[i+2][1] - PlusMinus_index[i+1][1])/2)
            angle = 0

            print('cutting_point_PlusMinus and angle', cutting_point, angle)

    print('angle', angle)
    #print('y_diff and frame_number:', y_diff, frame_number)
    #find the position in the middle of the segment with the lowest value and cut it
    if angle < 0.3: # threshold that might be important needs to be checked and adjusted

        print('final_cutting_point:', cutting_point)
        write_run_info('Core sample crosses the center and is cut at: ' + str(cutting_point))
        cut_frame1_poly = shapely.geometry.box(0, 0, cutting_point, imgheight)
        Multi_centerlines1= Multi_centerlines.intersection(cut_frame1_poly)
        cut_frame2_poly = shapely.geometry.box(cutting_point, 0, imgwidth, imgheight)
        Multi_centerlines2= Multi_centerlines.intersection(cut_frame2_poly)
        write_run_info("Multi_centerlines2 type {}".format(Multi_centerlines2.geom_type))
        measure_points1 = []
        measure_points2 = [] #I initiate it alredy here so i can use it in the test later
        #reorder Multi_centerlines1
        x_maxs = []
        x_mins = []
        for i in range(len(Multi_centerlines1)):
            minx, _, maxx,_ = Multi_centerlines1[i].bounds
            x_maxs.append(maxx)
            x_mins.append(minx)

        x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins))/2
        #print('x_middle, x_maxs, x_mins', x_middle, x_maxs, x_mins)
        contourszip = zip(x_middle, Multi_centerlines1)

        #print('contourszip', contourszip)
        #print('x_maxs', x_maxs)
        centerlines1 = [x for _,x in sorted(contourszip, key=itemgetter(0))]
        Multi_centerlines1 = shapely.geometry.MultiLineString(centerlines1)
        #print('ordered centerlines2:', Multi_centerlines2.geom_type)
        for i in range(len(Multi_centerlines1.geoms)-1):
            points = shapely.ops.nearest_points(Multi_centerlines1.geoms[i], Multi_centerlines1.geoms[i+1])
            measure_points1.append(points)

        if Multi_centerlines2.geom_type=='LineString':
            print("Multi_centerlines2 is only one line")
            write_run_info("Multi_centerlines2, the part after cutting point, is only one line")
        else:
            # order contours by x_maxs
            # Order by maxx
            x_maxs = []
            x_mins = []
            for i in range(len(Multi_centerlines2)):
                minx, _, maxx,_ = Multi_centerlines2[i].bounds
                x_maxs.append(maxx)
                x_mins.append(minx)

            x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins))/2
            contourszip = zip(x_middle, Multi_centerlines2)

            #print('contourszip', contourszip)
            #print('x_maxs', x_maxs)
            centerlines2 = [x for _,x in sorted(contourszip, key=itemgetter(0))]
            Multi_centerlines2 = shapely.geometry.MultiLineString(centerlines2)
            #print('ordered centerlines2:', Multi_centerlines2.geom_type)

            for i in range(len(Multi_centerlines2.geoms)-1):
                points = shapely.ops.nearest_points(Multi_centerlines2.geoms[i], Multi_centerlines2.geoms[i+1])
                measure_points2.append(points)

        if not measure_points2:
            measure_points=measure_points1
            Multi_centerlines = Multi_centerlines1
        else:
            measure_points=[measure_points1, measure_points2]
            Multi_centerlines = [Multi_centerlines1, Multi_centerlines2]

        return Multi_centerlines, measure_points, angle_index, cutting_point

    else:
        # loop through them to measure pairwise distances. if possible find nearest points and also visualise
        print('middle point was not detected')
        cutting_point = np.nan
        #reorder the lines
        x_maxs = []
        x_mins = []
        for i in range(len(Multi_centerlines)):
            minx, _, maxx,_ = Multi_centerlines[i].bounds
            x_maxs.append(maxx)
            x_mins.append(minx)

        x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins))/2
        contourszip = zip(x_middle, Multi_centerlines)

        #print('contourszip', contourszip)
        #print('x_maxs', x_maxs)
        centerlines = [x for _,x in sorted(contourszip, key=itemgetter(0))]
        Multi_centerlines = shapely.geometry.MultiLineString(centerlines)
        #print('ordered centerlines:', Multi_centerlines2.geom_type)
        measure_points = []
        for i in range(len(Multi_centerlines.geoms)-1):
            points = shapely.ops.nearest_points(Multi_centerlines.geoms[i], Multi_centerlines.geoms[i+1])
            measure_points.append(points)

        return Multi_centerlines, measure_points, angle_index, cutting_point

#######################################################################
# plot predicted lines and points of measurements to visually assess
#######################################################################
def plot_lines(image, centerlines, measure_points, file_name, angle_index, path_out):
    #create pngs folder in output path
    write_run_info("Plotting output as png")
    export_path = os.path.join(path_out, 'pngs')
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    #export_path = '/groups/swarts/lab/DendroImages/Plot8CNN_toTest/DetectionJpegs'

    #export_path = '/Users/miroslav.polacek/Documents/CNNRundomRubbish/CropTesting'
    f = file_name + '.png'

    #save images at original size unles they are bigger then 30000. Should improve diagnostics on the images
    imgheight, imgwidth = image.shape[:2]
    #sprint('imgheight, imgwidth', imgheight, imgwidth)
    plot_dpi = 100

    if imgwidth < 30000:
        plt.figure(figsize = (imgwidth/plot_dpi, 2*(imgheight/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
    else: #andjust image size if it`s exceeding 30000 pixels to 30000
        resized_height = imgheight*(30000/imgwidth)
        plt.figure(figsize = (30000/plot_dpi, 2*(resized_height/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)


    #plot the lines to the image
    if not isinstance(centerlines, list) and len(centerlines)>2:

        #plt.figure(figsize = (30,15))
        #plt.imshow(image)
        for i in range(len(centerlines.geoms)-1):
            #print('loop', i)
            points = measure_points[i]

            xc,yc = centerlines[i].coords.xy
            plt.plot(xc,yc,'g')

            xp, yp = points[0].coords.xy
            xp1, yp1 = points[1].coords.xy
            plt.plot([xp, xp1], [yp, yp1], 'r')

        xc,yc = centerlines[-1].coords.xy
        plt.plot(xc,yc,'g')
        #plt.show()


    elif isinstance(centerlines, list) and len(centerlines)==2:
        for l in range(2):
            color = ['g', 'b']
            centerlines1 = centerlines[l]
            #print('measure_points:', len(measure_points))
            measure_points1 = measure_points[l]
            if len(measure_points1)==0: # is only precoation in case the first part of measure points is empty
                continue
            for i in range(len(centerlines1.geoms)-1):
                #print('loop', i)

                xc,yc = centerlines1[i].coords.xy
                plt.plot(xc,yc,color[l])

                points = measure_points1[i]
                xp, yp = points[0].coords.xy
                xp1, yp1 = points[1].coords.xy
                plt.plot([xp, xp1], [yp, yp1], 'r')

            xc,yc = centerlines1[-1].coords.xy # to print the last point
            plt.plot(xc,yc, color[l])
        #plt.show()

    plt.savefig(os.path.join(export_path, f), bbox_inches = 'tight', pad_inches = 0)

#######################################################################
# create a json file for shiny
#######################################################################
def write_to_json(image_name, centerlines_rings, clean_contours_rings, clean_contours_cracks, cutting_point, run_ID, path_out):
    # define the structure of json
    ##pith_infered x,y, pith_from_circles,
    write_run_info("Writing .json file")
    out_json = {}
    out_json = {image_name: {'run_ID':run_ID, 'predictions':{}, 'directionality': {}, 'center': {}, 'est_rings_to_pith': {}, 'ring_widths': {}}} #directionality will be added in shiny, {x_value:{'width': VALUE , 'angle': VALUE},...}
    out_json[image_name]['predictions'] = {'ring_line': {}, 'ring_polygon': {}, 'crack_polygon': {}, 'resin_polygon': {}, 'pith_polygon': {}}
    out_json[image_name]['center'] = {'cutting_point': cutting_point, 'pith_present': np.nan, 'pith_inferred': {'coords': {'x': [], 'y': []}}} #pith_present is yes/no string if pith is on the image
    out_json[image_name]['ring_widths'] = {'directionality': {}, 'shortest_distance': {}, 'manual': {}} #each will have format 'undated_1'/year: {{(x1,y1),(x2,y2)},...}
    # separate x and y coordinates for polygons and line
    input_vars = [centerlines_rings, clean_contours_rings, clean_contours_cracks]
    json_names = ['ring_line', 'ring_polygon', 'crack_polygon']
    for v in range(len(input_vars)):
        coords = {}
        for i in range(len(input_vars[v])):

            # if else becasue ring_line is shapely object and clean contours are from opencv and have different structure
            if json_names[v] == 'ring_line':
                x_list, y_list = input_vars[v][i].coords.xy
                x_list = x_list.tolist()
                y_list = y_list.tolist()
            else:
                x_list = []
                y_list = []
                for p in range(len(input_vars[v][i])):
                    [[x,y]] = input_vars[v][i][p]
                    x_list.append(int(x))
                    y_list.append(int(y))
                #print("type(x_list)", type(x_list))
                #print("type(y_list)", type(y_list))
            x_min = math.floor(np.min(x_list))
            the_coord = str(x_min)+'_'+'coords'
            coords[the_coord] = {}
            coords[the_coord]['x'] = x_list
            coords[the_coord]['y'] = y_list
        #print("coords",type(coords))
        out_json[image_name]['predictions'][json_names[v]]=coords

    output = os.path.join(path_out, image_name.replace('.tif','.json'))
    with open(output,'w') as outfile:
        json.dump(out_json, outfile, indent=4)
#######################################################################
# create a .pos file with measure points
#######################################################################
def write_to_pos(centerlines, measure_points, file_name, image_name, DPI, path_out):
    print('Writing .pos file')
    write_run_info("Writing .pos file")
    # check if it is one or two parts in measure points
    #if two adjust naming. nothink for the normal one and may be some "x" at the end for the extra
    #print('measure_point len', len(measure_points))
    #print('measure_point', measure_points)
    #prepare date, time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    #prepare unit conversion
    pixel_per_mm = DPI/25.4
    #create paths for output files
    pos_name = file_name + '.pos'
    posX_name = file_name + 'X' + '.pos'
    out_file_path = os.path.join(path_out, pos_name)
    out_fileX_path = os.path.join(path_out, posX_name)
    #out_folder = '' #to output in the same file
    #print('names done')
    #print('centerline', centerlines)
    #print('len centrlines', type(centerlines))

    if not isinstance(centerlines, list) and len(centerlines)>2:
        #prepare points
        str_measure_points = []
        first_x, first_y = measure_points[0][0].coords.xy
        #print('first_xy', first_x, first_y)
        first_point = str(round(float(first_x[0])/pixel_per_mm, 3))+","+str(round(float(first_y[0])/pixel_per_mm, 3))
        str_measure_points.append(first_point)

        for i in range(len(measure_points)-1):
            #this gets second point of a current tuple and the first of the next tuple
            #print('measure_points', measure_points[i][0].coords.xy, measure_points[i][1].coords.xy)
            current_x, current_y = measure_points[i][1].coords.xy
            next_x, next_y = measure_points[i+1][0].coords.xy
            str_point = str(round(float(current_x[0])/pixel_per_mm, 3))+","+str(round(float(current_y[0])/pixel_per_mm,3))+"  "+str(round(float(next_x[0])/pixel_per_mm, 3))+","+str(round(float(next_y[0])/pixel_per_mm, 3))
            str_measure_points.append(str_point)
            #print('str_measure_points',str_measure_points)

        #print('should be last measure point', len(measure_points))
        last_x, last_y = measure_points[len(measure_points)-1][1].coords.xy
        last_point = str(round(float(last_x[0])/pixel_per_mm, 3))+","+str(round(float(last_y[0])/pixel_per_mm, 3))
        str_measure_points.append(last_point)

        #write in the file
        with open(out_file_path, 'w') as f:
            print('#DENDRO (Cybis Dendro program compatible format) Coordinate file written as', file=f)
            print('#Imagefile {}'.format(image_name), file=f)
            print('#DPI {}'.format(DPI), file=f)
            print('#All coordinates in millimeters (mm)', file=f)
            print('SCALE 1', file=f)
            print('#C DATED 2018', file=f) #not sure what this means and if it needst to be modified
            print('#C Written={};'.format(dt_string), file=f)
            print('#C CooRecorder=9.4 Sept 10 2019;', file=f) #what should we write here?
            print('#C licensedTo=Alexis Arizpe, alexis.arizpe@gmi.oeaw.ac.at;', file=f)
            for i in str_measure_points:
                print(i, file=f)

    elif isinstance(centerlines, list) and len(centerlines)==2:
        for l in range(2):

            measure_points1 = measure_points[l]
            print('len of measure_points1', len(measure_points1))
            if len(measure_points1)==0: # is only precoation in case the first part of measure points is empty
                write_run_info('Middle of the core identified on the first ring!!!Only X .pos file will be created!!!')
                continue
            str_measure_points1 = []
            first_x, first_y = measure_points1[0][0].coords.xy
            print('first_xy', first_x, first_y)
            first_point = str(round(float(first_x[0])/pixel_per_mm, 3))+","+str(round(float(first_y[0])/pixel_per_mm, 3))
            str_measure_points1.append(first_point)

            for i in range(len(measure_points1)-1):
                #this gets second point of a current tuple and the first of the next tuple
                #print('measure_points', measure_points[i][0].coords.xy, measure_points[i][1].coords.xy)
                current_x, current_y = measure_points1[i][1].coords.xy
                next_x, next_y = measure_points1[i+1][0].coords.xy
                str_point = str(round(float(current_x[0])/pixel_per_mm, 3))+","+str(round(float(current_y[0])/pixel_per_mm,3))+"  "+str(round(float(next_x[0])/pixel_per_mm, 3))+","+str(round(float(next_y[0])/pixel_per_mm, 3))
                str_measure_points1.append(str_point)
                #print('str_measure_points',str_measure_points1)

            print('should be last measure point', len(measure_points1))
            last_x, last_y = measure_points1[len(measure_points1)-1][1].coords.xy
            last_point = str(round(float(last_x[0])/pixel_per_mm, 3))+","+str(round(float(last_y[0])/pixel_per_mm, 3))
            str_measure_points1.append(last_point)

            #write in the file
            if l==0:
                with open(out_file_path, 'w') as f:
                    print('#DENDRO (Cybis Dendro program compatible format) Coordinate file written as', file=f)
                    print('#Imagefile {}'.format(image_name), file=f)
                    print('#DPI {}'.format(DPI), file=f)
                    print('#All coordinates in millimeters (mm)', file=f)
                    print('SCALE 1', file=f)
                    print('#C DATED 2018', file=f) #not sure what this means and if it needst to be modified
                    print('#C Written={};'.format(dt_string), file=f)
                    print('#C CooRecorder=9.4 Sept 10 2019;', file=f) #what should we write here?
                    print('#C licensedTo=Alexis Arizpe, alexis.arizpe@gmi.oeaw.ac.at;', file=f)
                    for i in str_measure_points1:
                        print(i, file=f)

            if l==1:
                with open(out_fileX_path, 'w') as f:
                    print('#DENDRO (Cybis Dendro program compatible format) Coordinate file written as', file=f)
                    print('#Imagefile {}'.format(image_name), file=f)
                    print('#DPI {}'.format(DPI), file=f)
                    print('#All coordinates in millimeters (mm)', file=f)
                    print('SCALE 1', file=f)
                    print('#C DATED 1000', file=f) #for now a 1000 later calculate the innermost ring from sampling date
                    print('#C Written={};'.format(dt_string), file=f)
                    print('#C CooRecorder=9.4 Sept 10 2019;', file=f) #what should we write here?
                    print('#C licensedTo=Alexis Arizpe, alexis.arizpe@gmi.oeaw.ac.at;', file=f)
                    for i in str_measure_points1:
                        print(i, file=f)

#######################################################################
# Run detection on an images
#######################################################################
path_out = os.path.join(args.output_folder, args.run_ID)
 # check if output dir for run_ID exists and if not create it
if not os.path.isdir(path_out):
    os.mkdir(path_out)

now = datetime.now()
dt_string_name = now.strftime("D%Y%m%d_%H%M%S") #"%Y-%m-%d_%H:%M:%S"
dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
run_ID = args.run_ID
log_file_name = 'CNN_' + run_ID + '_' + dt_string_name + '.log' #"RunID" + dt_string +
log_file_path =os.path.join(path_out, log_file_name)

# initiate log file
with open(log_file_path,"x") as fi:
    print("Run started:" + dt_string, file=fi)
    print("Ring weights used:" + weights_path_Ring, file=fi)
    print("Crack weights used:" + weights_path_Crack, file=fi)

# create a list of already exported jsons to prevent re-running the same image
json_list = []
for f in os.listdir(path_out):
    if f.endswith('.json'):
        json_name = f.replace('.json', '')
        json_list.append(json_name)

input = args.input
# check pathin if its folder or file and get file list of either
if os.path.isfile(input):
    # get file name and dir to file
    input_list = [os.path.basename(input)]
    input_path = os.path.split(input)[0]
elif os.path.isdir(input):
    # get a list of files in the dir
    input_list = os.listdir(input)
    input_path = input
else:
    print("Image argument is neither valid file nor directory")
    write_run_info("Image argument is neither valid file nor directory")
#print("got until here", input_list, input_path)

for f in input_list:
    if f.endswith('.tif') and f.replace('.tif', '') not in json_list:
        print("Processing image: {}".format(f))
        write_run_info("Processing image: {}".format(f))
        image_path = os.path.join(input_path, f)
        im_origin = skimage.io.imread(image_path)

        detected_mask_rings, detected_mask_cracks = sliding_window_detection(image = im_origin, overlap = 0.75, cropUpandDown = 0.17)
        print("detected_mask_rings", detected_mask_rings.shape)
        print("detected_mask_cracks", detected_mask_cracks.shape)
        clean_contours_rings = clean_up_mask(detected_mask_rings, is_ring=True)
        clean_contours_cracks = clean_up_mask(detected_mask_cracks, is_ring=False)

        centerlines_rings = find_centerlines(clean_contours_rings)

        centerlines, measure_points, angle_index, cutting_point = measure_contours(centerlines_rings, detected_mask_rings)

        write_to_json(f, centerlines_rings, clean_contours_rings, clean_contours_cracks, cutting_point, run_ID, path_out)
        image_name = f.replace('.tif', '')
        DPI = float(args.dpi)
        write_to_pos(centerlines, measure_points, image_name, f, DPI, path_out)

        # Ploting lines is moslty for debugging
        masked_image = im_origin.astype(np.uint32).copy()
        masked_image = apply_mask(masked_image, detected_mask_rings, alpha=0.2)
        masked_image = apply_mask(masked_image, detected_mask_cracks, alpha=0.3)
        plot_lines(masked_image, centerlines, measure_points, image_name, angle_index, path_out)
        write_run_info("IMAGE FINISHED")
