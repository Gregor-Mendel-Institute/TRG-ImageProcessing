"""
Load tiff image of whole core.
Run detections for squares in a form of sliding window with certain overlap to prevent problems of having ring directly in at the border.
Fuse all detections in one mask layer and consequently attach all of them to each other creating mask layer
of the size of original image. The detection confidance needs to be set in the config!
Print the image with mask over it.

FOR TESTING
conda activate TreeRingCNN &&
cd /Users/miroslav.polacek/Dropbox\ \(VBC\)/'Group Folder Swarts'/Research/CNNRings/Mask_RCNN/postprocessing &&
python3 detect_whole_core_measure_split_server.py --dpi=13039 --image=/Users/miroslav.polacek/Desktop/whole_core_examples/ --weight=/Users/miroslav.polacek/Dropbox\ \(VBC\)/Group\ Folder\ Swarts/Research/CNNRings/Mask_RCNN/logs/traintestmlw220200508T2011/mask_rcnn_traintestmlw2_0355.h5
 
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

parser.add_argument('--image', required=True,
                    metavar="/path/to/image/",
                    help="Path to image file")
parser.add_argument('--weight', required=True,
                    metavar="/path/to/weight/folder",
                    help="Path to weight file")
                    
parser.add_argument('--output_folder', required=True,
                    metavar="/path/to/weight/folder",
                    help="Path to weight file")

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
import time
import skimage
import pandas as pd
import numpy as np
import tensorflow as tf
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

from DetectionConfig import TreeRing

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = TreeRing.treeRingConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
weights_path = args.weight
print("Loading weights ")
model.load_weights(weights_path, by_name=True)

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
    out_dir = args.output_folder
    log_files = []
    for f in os.listdir(out_dir):
        if f.startswith("CNN_"):
            log_files.append(f)
            
    log_file_name = os.path.join(out_dir, log_files[-1])       
    with open(log_file_name,"a") as f:
        print(string, file=f)


############################################################################################################
# Sliding window detection with rotation of each part of image by 90 and 45 degrees and combining the output

############################################################################################################

def sliding_window_detection(image, overlap = 0.5):
    #print('im before pad', im_origin.shape)
    imgheight_for_pad, imgwidth_for_pad = im_origin.shape[:2]
    #print('image', im.shape)
    # add zero padding at the begining and the end according to overlap so every part of the picture is detected same number of times
    zero_padding_front = np.zeros(shape=(imgheight_for_pad, int(imgheight_for_pad-(imgheight_for_pad*overlap)),3))
    zero_padding_back = np.zeros(shape=(imgheight_for_pad, imgheight_for_pad,3))
    #print('padding', zero_padding.shape)
    im_padded = np.concatenate((zero_padding_front, im_origin, zero_padding_back), axis=1)

    imgheight, imgwidth = im_padded.shape[:2]
    #print('im_after_pad', im_padded.shape)

    the_mask = np.zeros(shape=(imgheight, imgwidth)) #combine all the partial masks in the final size of full tiff
    #print('the_mask', the_mask.shape)
    looping_list = range(0,imgwidth, int(imgheight-(imgheight*overlap)))
    cut_end = int(zero_padding_back.shape[1]/int(imgheight-(imgheight*overlap)))
    #print('cut_end', cut_end)
    for i in looping_list[:-cut_end]: #defines the slide value
        # crop the image
        cropped_part = im_padded[:imgheight, i:(i+imgheight)]
        #print('cropped_part', cropped_part.shape)
        # run detection on the cropped part of the image
        results = model.detect([cropped_part], verbose=0)
        r = results[0]
        #visualize.display_instances(cropped_part, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']) #just to check

        # rotate image 90 and run detection
        cropped_part_90 = skimage.transform.rotate(cropped_part, 90, preserve_range=True).astype(np.uint8)
        results1 = model.detect([cropped_part_90], verbose=0)
        r1 = results1[0]
        #visualize.display_instances(cropped_part_90, r1['rois'], r1['masks'], r1['class_ids'], class_names, r1['scores']) #just to check

        # rotate image 45 and run detection
        cropped_part_45 = skimage.transform.rotate(cropped_part, angle = 45, resize=True).astype(np.uint8)
        results2 = model.detect([cropped_part_45], verbose=0)
        r2 = results2[0]

        ## flatten all in one layer. This should be one layer of zeroes and ones. If applying som e cleaning i would do it before this point or cleaning of the final one might be even better
        maskr = np.zeros(shape=(imgheight, imgheight))
        #print('maskr_empty', maskr)
        nmasks = r['masks'].shape[2]
        #print(nmasks)
        for m in range(0,nmasks):
            maskr = maskr + r['masks'][:,:,m]
            #print(maskr.sum())

        maskr1 = np.zeros(shape=(imgheight, imgheight))
        nmasks1 = r1['masks'].shape[2]
        for m in range(0,nmasks1):
            maskr1 = maskr1 + r1['masks'][:,:,m]
        ## rotate maskr1 masks back
        maskr1_back = np.rot90(maskr1, k=-1)
        # beware different dimensions!!!

        imheight2 = r2['masks'].shape[0]
        nmasks2 = r2['masks'].shape[2]
        maskr2 = np.zeros(shape=(imheight2, imheight2))
        for m in range(0,nmasks2):
            maskr2 = maskr2 + r2['masks'][:,:,m]
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
        #print('middle', all_masks_combined.shape)
        end_the_mask = the_mask[:imgheight, (i+imgheight):]

        if i == 0: # to solve the begining
            the_mask = np.concatenate((all_masks_combined, end_the_mask), axis=1)
        else:
            begining_the_mask = the_mask[:imgheight, :i]
            the_mask = np.concatenate((begining_the_mask, all_masks_combined, end_the_mask), axis=1)


    # First remove the padding
    pad_front = zero_padding_front.shape[1]
    #print('front', pad_front)
    pad_back = zero_padding_back.shape[1]
    the_mask_clean = the_mask[:,pad_front:-pad_back]
    #print('the_mask', the_mask.shape)
    #print('the_mask_clean', the_mask_clean.shape)
    #plt.imshow(the_mask_clean) # uncomment to print mask layer
    #plt.show()
    # TO PRINT THE MASK OVERLAYING THE IMAGE

    return the_mask_clean

#######################################################################
# Extract distances from the mask
#######################################################################

def clean_up_mask(mask):
    # detects countours of the masks, removes small contours, fits circle to individual contours and estimates the pith, skeletonizes the detected contours

    # make the mask binary
    binary_mask = np.where(mask >= 1, 255, 0) # this part can be cleaned to remove some missdetections setting condition for >2
    #plt.imshow(binary_mask)
    #plt.show()
    type(binary_mask)
    uint8binary = binary_mask.astype(np.uint8).copy()

    #gray_image = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    image, contours, hierarchy = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print('contour_shape:', len(contours))

    #### here i extract dimensions and angle of individual contours bigger than threshold
    contours_filtered = []
    contour_lengths = []
    imgheight, imgwidth = mask.shape[:2]
    for i in range(0, len(contours)): #len(contours)):
        #remove those that are too short
        rect = cv2.minAreaRect(contours[i])
        #print(rect)
        imgheight, imgwidth = mask.shape[:2]
        #print(imgheight)
        dim1, dim2 = rect[1]
        dim_max = max([dim1, dim2])
        dim_sum = dim1 + dim2 # should capture size and curvature better then just a length
        if dim_max > imgheight/5: #will take only contours that are bigger than 1/5 of the image
            contours_filtered.append(contours[i])

            contour_lengths.append(dim_sum)

    #print('Filtered_contours:', len(contours_filtered))
    #print('contour_lengths:', contour_lengths)
    #print(contours_filtered[0])

    #### Extract longest contour to use for center estimate
    contourszip = zip(contour_lengths, contours_filtered)

    ordered_contours = [x for _,x in sorted(contourszip, reverse = True)]
    #print('ordered:', ordered_contours)
    # in final should return cleaned contours and center coordinates
    #return  centroide_ordered10, centroide_ordered5, centroide_ordered2, centroide_ordered1
    return ordered_contours # Returns filtered and orderedt contours

#######################################################################
# Turn contours into lines and find nearest points between them for measure
#######################################################################
# return table of distances or paired point coordinates?
def measure_contours(clean_contours, image):
    # find center line of contours and find nearest_points for each pair of lines
    ##first need to reorganise the data
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
        print('ring_contour:', i)
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
    px_to_cut_off = 200 #based on examples that i used for testing
    Multi_centerlines_to_crop = shapely.geometry.MultiLineString(centerlines)
    minx, miny, maxx, maxy = Multi_centerlines_to_crop.bounds
    #print('minx, miny, maxx, maxy', minx, miny, maxx, maxy)
    frame_to_crop = shapely.geometry.box(minx, miny+px_to_cut_off, maxx, maxy-px_to_cut_off)
    Multi_centerlines = Multi_centerlines_to_crop.intersection(frame_to_crop)
    # to check if it cropps something
    minx, miny, maxx, maxy = Multi_centerlines.bounds
    #print('minx, miny, maxx, maxy after', minx, miny, maxx, maxy)

    ## Split samples that are crosing center into two then turn the other part around
    #may be I need image dimensions
    # try to use shapely see shapely intersection
    imgheight, imgwidth = image.shape[:2]
    write_run_info("Image has height {} and width {}".format(imgheight, imgwidth))
    write_run_info("{} ring boundries were detected".format(len(centerlines)))
    angle_index = [] #average angle should work. Then
    frame_width = 500
    number_of_segments = int(imgwidth/frame_width)
    #plt.imshow(image)
    for i in range(0, number_of_segments):
        #get the frame
        frame_poly = shapely.geometry.box(i*frame_width, 0, (i*frame_width)+frame_width, imgheight)
        #print(frame_poly.exterior.coords.xy)
        #x, y = frame_poly.exterior.coords.xy
        #plt.plot(x,y)
        #get lines inside of the frame
        intersection = Multi_centerlines.intersection(frame_poly)
        #print('intersection:', intersection.geom_type)
        if intersection.geom_type=='LineString':
            line_coords = intersection.coords
            line_coords = sorted(line_coords, reverse = True)
            #print('sorted:', line_coords)
            x0, y0 = line_coords[0]
            xlast, ylast = line_coords[-1]
            #solve problem with short peaces
            x_dif = abs(xlast - x0)
            if x_dif < frame_width:
                #print(i, 'th is too short')
                continue
            else:
                y_dif = abs(y0 - ylast)
                angle_index.append([y_dif, i]) #I add also i here so i can identify distance later
                #print('linestring_x',x_int)
                #plt.plot(x_int, y_int)

        elif intersection.geom_type=='MultiLineString':
            y_difs = []
            for l in range(len(intersection.geoms)):
                line_coords = intersection.geoms[l].coords
                #x_int, y_int = line_coords.xy
                line_coords = sorted(line_coords, reverse = True)
                #print('sorted:', line_coords)
                x0, y0 = line_coords[0]
                xlast, ylast = line_coords[-1]
                y_dif = abs(y0 - ylast)
                y_difs.append(y_dif)
                #print('Multi_x:', x_int)
                #plt.plot(x_int, y_int)
            angle_index.append([np.mean(y_difs), i])

        else:
            #print('second else????')
            continue

    print('angle_index:', angle_index)
    #now find the one with minimal value and find which frame it was based on i
    y_diff, frame_number = sorted(angle_index)[0]
    #print('y_diff and frame_number:', y_diff, frame_number)
    #find the position in the middle of the segment with the lowest value and cut it
    if y_diff < 50: # threshold that might be important needs to be checked and adjusted 100 seems ok
        cutting_point = (frame_number*frame_width)+(frame_width/2)
        print('cutting_point:', cutting_point)
        write_run_info('Core sample crosses the center and is cut at: ' + str(cutting_point))
        cut_frame1_poly = shapely.geometry.box(0, 0, cutting_point, imgheight)
        Multi_centerlines1= Multi_centerlines.intersection(cut_frame1_poly)
        cut_frame2_poly = shapely.geometry.box(cutting_point, 0, imgwidth, imgheight)
        Multi_centerlines2= Multi_centerlines.intersection(cut_frame2_poly)
        write_run_info("Multi_centerlines2 type {}".format(Multi_centerlines2.geom_type))
        measure_points1 = []
        measure_points2 = [] #I initiate it alredy here so i can use it in the test later
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
            for i in range(len(Multi_centerlines2)):
                _, _, maxx,_ = Multi_centerlines2[i].bounds
                x_maxs.append(maxx)

            contourszip = zip(x_maxs, Multi_centerlines2)

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

        return Multi_centerlines, measure_points


    else:
        # loop through them to measure pairwise distances. if possible find nearest points and also visualise
        measure_points = []
        for i in range(len(Multi_centerlines.geoms)-1):
            points = shapely.ops.nearest_points(Multi_centerlines.geoms[i], Multi_centerlines.geoms[i+1])
            measure_points.append(points)
        return Multi_centerlines, measure_points
#######################################################################
# Estimate pith position
#######################################################################
"""
def estimate_pith(Multi_centerlines):
    #### Try to find the circle based on all detected boundreis

    x_centers = []
    y_centers = []
    for i in range(0, len(contours_filtered)):

        x_cir = []
        y_cir = []
        for p in range(0, len(contours_filtered[i])):
            [[xs, ys]] = contours_filtered[i][p]
            #print("xs", xs)
            x_cir.append(xs)
            y_cir.append(ys)

        data = list(zip(x_cir,y_cir))
        xc,yc,r,_ = cf.hyper_fit(data)
        #print('hyper_fit:', xc, yc, r)
        x_centers.append(xc)
        y_centers.append(yc)
    centroide = (np.mean(x_centers),np.mean(y_centers))
    print(centroide)
"""

#######################################################################
# plot predicted lines and points of measurements to visually assess
#######################################################################
def plot_lines(image, centerlines, measure_points, file_name):
    #create pngs folder in output path
    export_path = os.path.join(args.output_folder, 'pngs')
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    
    #export_path = '/groups/swarts/lab/DendroImages/Plot8CNN_toTest/DetectionJpegs'
    
    #export_path = '/Users/miroslav.polacek/Documents/CNNRundomRubbish/CropTesting'
    f = file_name + '.png'

    imgheight, imgwidth = image.shape[:2]
    plot_dpi = 600
    plt.figure(figsize = (13, 4), dpi=plot_dpi)
    plt.imshow(image)

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
            print('measure_points:', len(measure_points))
            measure_points1 = measure_points[l]
            for i in range(len(centerlines1.geoms)-1):
                #print('loop', i)
                points = measure_points1[i]

                xc,yc = centerlines1[i].coords.xy
                plt.plot(xc,yc,color[l])

                xp, yp = points[0].coords.xy
                xp1, yp1 = points[1].coords.xy
                plt.plot([xp, xp1], [yp, yp1], 'r')

            xc,yc = centerlines1[-1].coords.xy
            plt.plot(xc,yc, color[l])
        #plt.show()
    plt.savefig(os.path.join(export_path, f))
#######################################################################
# create a .pos file with measure points
#######################################################################
def write_to_pos(centerlines, measure_points, file_name, image_name, DPI):
    print('Writing .pos file')
    # check if it is one or two parts in measure points
    #if two adjust naming. nothink for the normal one and may be some "x" at the end for the extra
    #print('measure_point len', len(measure_points))
    #print('measure_point', measure_points)
    #prepare date, time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    #prepare unit conversion
    pixel_per_mm = DPI/25.4
    out_folder = args.output_folder
    #out_folder = '' #to output in the same file


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
        with open(out_folder + file_name+'.pos', 'w') as f:
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

            str_measure_points1 = []
            first_x, first_y = measure_points1[0][0].coords.xy
            #print('first_xy', first_x, first_y)
            first_point = str(round(float(first_x[0])/pixel_per_mm, 3))+","+str(round(float(first_y[0])/pixel_per_mm, 3))
            str_measure_points1.append(first_point)

            for i in range(len(measure_points1)-1):
                #this gets second point of a current tuple and the first of the next tuple
                #print('measure_points', measure_points[i][0].coords.xy, measure_points[i][1].coords.xy)
                current_x, current_y = measure_points1[i][1].coords.xy
                next_x, next_y = measure_points1[i+1][0].coords.xy
                str_point = str(round(float(current_x[0])/pixel_per_mm, 3))+","+str(round(float(current_y[0])/pixel_per_mm,3))+"  "+str(round(float(next_x[0])/pixel_per_mm, 3))+","+str(round(float(next_y[0])/pixel_per_mm, 3))
                str_measure_points1.append(str_point)
                #print('str_measure_points',str_measure_points)

            #print('should be last measure point', len(measure_points1))
            last_x, last_y = measure_points1[len(measure_points1)-1][1].coords.xy
            last_point = str(round(float(last_x[0])/pixel_per_mm, 3))+","+str(round(float(last_y[0])/pixel_per_mm, 3))
            str_measure_points1.append(last_point)

            #write in the file
            if l==0:
                with open(out_folder + file_name+'.pos', 'w') as f:
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
                with open(out_folder + file_name+'X'+'.pos', 'w') as f:
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
# Run detection on an image
#######################################################################
#initiate run information and create the log file in outpout dir
now = datetime.now()
dt_string_name = now.strftime("%Y%m%d_%H") #"%Y-%m-%d_%H:%M:%S"
dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
file_name = 'CNN_' + dt_string_name + '.log' #"RunID" + dt_string +
file_path =os.path.join(args.output_folder, file_name)

with open(file_path,"x") as fi:
    print("Run started:" + dt_string, file=fi)

pathpos = args.output_folder
pos_list = []
for f in os.listdir(pathpos):
    if f.endswith('.pos'):
        pos_name = f.split('.')[0]
        pos_list.append(pos_name)

pathin = args.image
for f in os.listdir(pathin):
    if f.endswith('.tif') and f.split('.')[0] not in pos_list:
        print(f)
        write_run_info("Processing image: {}".format(f))
        image_path = os.path.join(pathin, f)
        image_name = os.path.basename(image_path)
        im_origin = skimage.io.imread(image_path)
        try:
            detected_mask = sliding_window_detection(image = im_origin, overlap = 0.5)
        except Exception as e:
            write_run_info("Function sliding_window_detection failed: {}".format(e))
        try:
            clean_contours = clean_up_mask(detected_mask)
        except Exception as e:
            write_run_info("Function clean_up_mask failed: {}".format(e))
        try:
            centerlines, measure_points = measure_contours(clean_contours, detected_mask)
        except Exception as e:
            write_run_info("Function measure_contours failed: {}".format(e))

        file_name = os.path.basename(image_path).split('.')[0]
        print('file_name', file_name)
        DPI = float(args.dpi)
        #PithCoordinates = 'none' # NOT YET IMPLEMENTED
        try:
            write_to_pos(centerlines, measure_points, file_name, image_name, DPI)
        except Exception as e:
            write_run_info("Function write_to_pos failed: {}".format(e))

        # Ploting lines is moslty for debugging
        masked_image = im_origin.astype(np.uint32).copy()
        masked_image = apply_mask(masked_image, detected_mask, alpha=0.2)
        plot_lines(masked_image, centerlines, measure_points, file_name)
# develop this to run on all images in a folder

"""
# this part can be commented now since i saved the mask separately to save time
for f in os.listdir(pathin):
    if f.endswith('.tif'):
        print(f)
        image_path = os.path.join(pathin, f)
        im_origin = skimage.io.imread(image_path)
        detected_mask = sliding_window_detection(image = im_origin, overlap = 0.5)
        #to_save_mask = '/Users/miroslav.polacek/Desktop/whole_core_examples/mask_example/detected_mask.npy'
        #np.save(to_save_mask, detected_mask)
        masked_image = im_origin.astype(np.uint32).copy()
        masked_image = apply_mask(masked_image, detected_mask, alpha=0.2)
        clean_contours = clean_up_mask(detected_mask)
        centerlines, measure_points = measure_contours(clean_contours)
        print('centerlines:', len(centerlines))
        print('measure_points:', len(measure_points))
        plot_lines(masked_image, centerlines, measure_points)


        centroide_ordered10, centroide_ordered5, centroide_ordered2, centroide_ordered1 = clean_up_mask(detected_mask)

        masked_image = im_origin.astype(np.uint32).copy()
        masked_image = apply_mask(masked_image, detected_mask, alpha=0.3)

        plt.imshow(masked_image)
        plt.plot(centroide_ordered1[0], centroide_ordered1[1], 'ro', markersize=5)
        plt.plot(centroide_ordered2[0], centroide_ordered2[1], 'go', markersize=5)
        plt.plot(centroide_ordered5[0], centroide_ordered5[1], 'bo', markersize=5)
        plt.plot(centroide_ordered10[0], centroide_ordered10[1], 'co', markersize=5)
        #plt.plot(centroide_quantiles[0], centroide_quantiles[1], 'mo', markersize=5)
        #plt.plot(centroide_short_distances[0], centroide_short_distances[1], 'yo', markersize=5)
        #plt.show()
        export_path = '/Users/miroslav.polacek/Documents/CNNRundomRubbish/20200424_detections/MLW2_conf95_medians'
        plt.savefig(os.path.join(export_path, f), dpi=300)

"""
"""
# for testing to load saved mask

mask_path = '/Users/miroslav.polacek/Desktop/whole_core_examples/mask_example/detected_mask.npy'
detected_mask = np.load(mask_path)
print('mask_shape:', detected_mask.shape)

clean_contours = clean_up_mask(detected_mask)
centerlines, measure_points = measure_contours(clean_contours, detected_mask)
print('centerlines:', len(centerlines))
print('measure_points:', len(measure_points))
image_name = "example_name"
plot_lines(detected_mask, centerlines, measure_points, image_name)

#file_name = 'testpos'
#image = 'sometif'
#DPI = 13039.0

#write_to_pos(centerlines, measure_points, file_name, image, DPI)
"""
"""
# just to test the functions to fit the circle
x = [36, 36, 19, 18, 33, 26]
y = [14, 10, 28, 31, 18, 26]
data = list(zip(x,y))
print(data)
xc,yc,r,_ = cf.least_squares_circle((data))
print('circle_least_square:', xc, yc, r)

xchf,ychf,rhf,_ = cf.hyper_fit((data))
print('circle_hyper_fit:', xchf, ychf, rhf)
"""
#plt.imshow(skelet_mask)
#plt.show()


        #masked_image = im_origin.astype(np.uint32).copy()
        #masked_image = apply_mask(masked_image, detected_mask, alpha=0.3)

        #plt.imshow(masked_image) # uncomment to print masked image
        #plt.show()

        #export_path = '/Users/miroslav.polacek/Documents/CNNRundomRubbish/200320_longcoredetections/newaug_E239_conf_0.9'
        #plt.savefig(os.path.join(export_path, f), dpi=300)
            # Add small mask to the_mask with the shift of i

            #from skimage.morphology import skeletonize
            #skeleton = skeletonize(image)


# Only for testing the extract_distances

# which should run when run in console
#if __name__ == "__main__": # i think not necessary at the moment
#    sliding()
