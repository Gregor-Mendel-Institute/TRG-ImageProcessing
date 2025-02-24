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
# Imports and prepare model
#######################################################################
import os
import sys
import math
import cv2
import json
import time
import argparse
import skimage
import skimage.io
import pickle
from skimage import exposure, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import box
from shapely.ops import nearest_points
import scipy
from datetime import datetime
from operator import itemgetter

# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
#print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.src_get_centerline import get_centerline
import mrcnn.model as modellib
from DetectionConfig import TreeRing_onlyRing
from DetectionConfig import TreeRing_onlyCracks
from training.retraining_container import retraining
from training.prepareAnnotations import prepareAnnotations

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
    parser.add_argument('--weightCrack', required=False,
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

    parser.add_argument('--start_new', required=False,
                        default="True",
                        help='If True retraining wil start from the beginning else continue from provided weight')

    args = parser.parse_args()
    return args
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
# Write run information
#######################################################################
# Should find the log file and add data to it
def write_run_info(string):
    # Get name of the log file in the output dir
    try: # in a bit of desperation
        out_dir = os.path.join(args.output_folder, args.run_ID)
        run_ID = args.run_ID
        log_files = []
        for f in os.listdir(out_dir):
            if f.startswith(str(args.logfile) + run_ID):
                path_f = os.path.join(out_dir, f)
                log_files.append(path_f)

        # Sort files by creation times
        log_files.sort(key=os.path.getctime)
        #print("sorted log files list", log_files)

        log_file_name = log_files[-1]
        with open(log_file_name,"a") as f:
            print(string, file=f)
    except:
        pass
############################################################################################################
# Sliding window detection with rotation of each part of image by 90 and 45 degrees and combining the output
############################################################################################################
def sliding_window_detection_multirow(image, detection_rows=1, modelRing=None, modelCrack=None, overlap=0.75, row_overlap=0.1, cropUpandDown=0.17):
    print("sliding_window_detection_multirow started")
    write_run_info("Sliding window overlap = {} and cropUpandDown = {}".format(overlap, cropUpandDown))
    # Crop image top and bottom to avoid detectectig useles part of the image
    imgheight_origin, imgwidth_origin = image.shape[:2]

    #print('image shape', image.shape[:2])
    #print('cropUpandDown', cropUpandDown)
    to_crop = int(imgheight_origin*cropUpandDown)
    new_image = image[to_crop:(imgheight_origin-to_crop), :, :]
    #print('new image shape', new_image.shape)

    imgheight_for_pad, imgwidth_for_pad = new_image.shape[:2]

    # add zero padding at the begining and the end according to overlap so every part of the picture is detected same number of times
    zero_padding_front = np.zeros(shape=(imgheight_for_pad, int(imgheight_for_pad*overlap),3), dtype=int)
    zero_padding_back = np.zeros(shape=(imgheight_for_pad, imgheight_for_pad,3), dtype=int)
    #print('padding', zero_padding.shape)
    im_padded = np.concatenate((zero_padding_front, new_image, zero_padding_back), axis=1)

    imgheight, imgwidth = im_padded.shape[:2]
    #print('im_after_pad', im_padded.shape)

    # Define sliding window parameters
    ## rows
    if detection_rows > 1:
        row_overlap = row_overlap
        row_overlap_height = int((imgheight*row_overlap)/(detection_rows-1))
        row_height = int(((row_overlap_height*(detection_rows-1))+imgheight)/detection_rows)
        row_looping_range = range(0, imgheight-(row_height-1), int(row_height-row_overlap_height))
    else:
        row_looping_range = [0]
        row_height = imgheight

    ## columns
    looping_range = range(0,imgwidth, int(row_height-(row_height*overlap)))
    looping_list = [i for i in looping_range if i < int(row_height-(row_height*overlap)) + imgwidth_origin]
    #print('looping_list', looping_list)

    combined_masks_per_class = np.empty(shape=(imgheight, imgwidth,0))
    if modelCrack==None:
        models = [modelRing]
    else:
        models = [modelRing, modelCrack]

    for model in models:
        the_mask = np.zeros(shape=(imgheight, imgwidth), dtype=int) # combine all the partial masks in the final size of full tiff
        #print('the_mask', the_mask.shape)
        for rl in row_looping_range:
            #print("r", rl)
            for i in looping_list: # defines the slide value
                #print("i", i)

                #print("i", i)
                # crop the image
                cropped_part = im_padded[rl:rl+row_height, i:i+row_height]
                #print('cropped_part, i, i+imheight', cropped_part.shape, i, i+imgheight)

                # Run detection on the cropped part of the image
                results = model.detect([cropped_part], verbose=0)
                r = results[0]
                r_mask = r['masks']
                #visualize.display_instances(cropped_part, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']) #just to check

                # Rotate image 90 and run detection
                cropped_part_90 = skimage.transform.rotate(cropped_part, angle=90, preserve_range=True).astype(np.uint8)
                results1 = model.detect([cropped_part_90], verbose=0)
                r1 = results1[0]
                r1_mask = r1['masks']
                #visualize.display_instances(cropped_part_90, r1['rois'], r1['masks'], r1['class_ids'], class_names, r1['scores']) #just to check

                # Rotate image 45 and run detection
                cropped_part_45 = skimage.transform.rotate(cropped_part, angle=45,
                                                           preserve_range=True, resize=True).astype(np.uint8)
                results2 = model.detect([cropped_part_45], verbose=0)
                r2 = results2[0]
                r2_mask = r2['masks']

                ## Flatten all in one layer
                maskr = np.zeros(shape=(row_height, row_height), dtype=int)
                #print('maskr_empty', maskr)
                nmasks = r_mask.shape[2]
                #print(nmasks)
                for m in range(0,nmasks):
                    maskr = maskr + r_mask[:,:,m]
                    #print(maskr.sum())

                maskr1 = np.zeros(shape=(row_height, row_height), dtype=int)
                nmasks1 = r1_mask.shape[2]
                for m in range(0,nmasks1):
                    maskr1 = maskr1 + r1_mask[:,:,m]
                # Rotate maskr1 masks back
                maskr1_back = np.rot90(maskr1, k=-1)
                # Beware different dimensions!!!

                imheight2 = r2_mask.shape[0]
                nmasks2 = r2_mask.shape[2]
                maskr2 = np.zeros(shape=(imheight2, imheight2), dtype=int)
                for m in range(0,nmasks2):
                    maskr2 = maskr2 + r2_mask[:,:,m]
                # Rotate back
                maskr2_back = skimage.transform.rotate(maskr2, angle=-45, resize=False)
                # Crop to the right size
                to_crop = int((imheight2 - row_height)/2)

                maskr2_back_cropped = maskr2_back[to_crop:(to_crop+int(row_height)), to_crop:(to_crop+int(row_height))]

                # Put all togather
                combined_mask = maskr1_back + maskr + maskr2_back_cropped
                #print("combined mask shape", combined_mask.shape)
                #print("the_mask shape", the_mask.shape)
                the_mask[rl:rl+row_height, i:i+row_height] = the_mask[rl:rl+row_height, i:i+row_height] + combined_mask

        the_mask = np.reshape(the_mask, (the_mask.shape[0],the_mask.shape[1],1))
        combined_masks_per_class = np.append(combined_masks_per_class, the_mask, axis=2)

    # First remove the padding
    pad_front = zero_padding_front.shape[1]
    #print('front', pad_front)
    pad_back = zero_padding_back.shape[1]
    the_mask_clean = combined_masks_per_class[:,pad_front:-pad_back,:]
    #print('the_mask_clean.shape', the_mask_clean.shape)


    # Concatanete the top and buttom to fit the original image
    missing_part = int((imgheight_origin - the_mask_clean.shape[0])/2)
    to_concatenate = np.zeros(shape=(missing_part, imgwidth_origin, the_mask_clean.shape[2]), dtype=int)
    #print("to_concatenate", to_concatenate.shape)
    the_mask_clean_origin_size = np.concatenate((to_concatenate, the_mask_clean, to_concatenate),axis=0)
    #print('the_mask_clean_origin_size', the_mask_clean_origin_size.shape)
    #plt.imshow(the_mask_clean) # uncomment to print mask layer
    #plt.show()

    # The mask for ring is in position the_mask_clean_origin_size[:,:,0] while cracks in the_mask_clean_origin_size[:,:,1]
    return the_mask_clean_origin_size

#######################################################################
# Extract distances from the mask
#######################################################################
def clean_up_mask(mask, min_mask_overlap=3, is_ring=True):
    # Detects countours of the masks, removes small contours
    print("clean_up_mask started")
    # Make the mask binary
    binary_mask = np.where(mask >= min_mask_overlap, 255, 0) # this part can be cleaned to remove some missdetections setting condition for higher value
    #print("binary_mask shape", binary_mask.shape)
    #plt.show()
    #type(binary_mask)
    uint8binary = binary_mask.astype(np.uint8).copy()

    # Older version of openCV has slightly different syntax I adjusted for it here
    if int(cv2.__version__.split(".")[0]) < 4:
        _, contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print('contour_shape:', len(contours))

    # Here i extract dimensions and angle of individual contours bigger than threshold
    imgheight, imgwidth = mask.shape[:2]
    if is_ring==True:
        min_size_threshold = imgheight/5 # Will take only contours that are bigger than 1/5 of the image
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
        dim_sum = dim1 + dim2 # Should capture size and curvature better then just a length
        if dim_max > min_size_threshold:
            contours_filtered.append(contours[i])
            x_mins.append(x_min)
            print("contour shape", contours[i].shape)

    print('Filtered_contours:', len(contours_filtered))

    #print(contours_filtered[0])

    # Extract longest contour to use for center estimate
    if is_ring==True:
        unique_vector = range(len(x_mins)) # added to prevent sorting problems if x_mins values are the same
        contourszip = zip(x_mins, unique_vector, contours_filtered)
        contours_out = [x for _,_, x in sorted(contourszip, reverse=False)]
    else:
        contours_out = contours_filtered

    # Returns filtered and ordered contours
    return contours_out

#######################################################################
# Finds centerlines in contours
#######################################################################
def find_centerlines(clean_contours, cut_off=0.01, y_length_threshold=100):
    # Find ceneterlines in polygons
    # cut_off clips upper and lower edges which are sometimes turning horizontal and affect measurements
    # y_length_threshold removes lines that are too short on y axes thus most probably horizontal misdetections
    print("find_centerlines started")
    # First need to reorganise the data
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

    # Order contours by x_min
    contourszip = zip(x_mins, contours_tuples)
    contours_tuples = [x for _,x in sorted(contourszip, key=itemgetter(0))]

    centerlines = []
    for i in range(len(contours_tuples)):
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
        #print("miny and maxy", miny, maxy)
        #print("line_y_diff", line_y_diff)
        if line_y_diff < y_length_threshold: # This threshold is in px. Originaly 100
            continue
        else:
            centerlines.append(cline)

    # test if centerline list contains something and if not abort and give a message
    if not centerlines: # empty list is False
        print("NO LINES LEFT AFTER CLEANING")
        write_run_info("NO LINES LEFT AFTER CLEANING")
        write_run_info("One reason could be that your images have too much background."
                       "Ideally, there should not be too much background above and below the core."
                       "Try to crop tighter.")
        return
    else:
        ## Cut off upper and lower part of detected lines. It should help with problems of horizontal ends of detections
        Multi_centerlines_to_crop = shapely.geometry.MultiLineString(centerlines)
        minx, miny, maxx, maxy = Multi_centerlines_to_crop.bounds
        px_to_cut_off = (maxy-miny)*cut_off
        #print('minx, miny, maxx, maxy', minx, miny, maxx, maxy)
        frame_to_crop = shapely.geometry.box(minx, miny+px_to_cut_off, maxx, maxy-px_to_cut_off)
        Multi_centerlines = Multi_centerlines_to_crop.intersection(frame_to_crop)
        # To check if it cropps something
        #minx, miny, maxx, maxy = Multi_centerlines.bounds
        #print('minx, miny, maxx, maxy after', minx, miny, maxx, maxy)

    return Multi_centerlines

#######################################################################
# Turn contours into lines and find nearest points between them for measure
#######################################################################
# Return table of distances or paired point coordinates
def measure_contours(Multi_centerlines, image):
    print("measure_contours started")
    imgheight, imgwidth = image.shape[:2]
    print('imgheight, imgwidth', imgheight, imgwidth)
    write_run_info("Image has height {} and width {}".format(imgheight, imgwidth))
    write_run_info("{} ring boundries were detected".format(len(Multi_centerlines)))

    # Split samples that are crosing center into two then turn the second part around
    PlusMinus_index = []
    frame_width = imgheight * .75
    sliding = frame_width * .5 # How much is the frame sliding in every frame
    #print('frame_width', frame_width)
    number_of_segments = int(imgwidth/sliding)
    #print('number_of_segments', number_of_segments)
    #plt.imshow(image)
    for i in range(0, number_of_segments):
        #print('loop_number', i)
        # get the frame
        frame_poly = shapely.geometry.box(i*sliding, 0, (i*sliding)+frame_width, imgheight)
        cut_point = i*sliding+(frame_width*.5) # Better to get cutting point here and use instead of frame number
        #print('cutting_point', cutting_point)
        #print('frame_exterior_xy',frame_poly.exterior.coords.xy)
        #x, y = frame_poly.exterior.coords.xy
        #plt.plot(x,y)
        # get lines inside of the frame
        intersection = Multi_centerlines.intersection(frame_poly)
        #print('intersection:', intersection.geom_type)
        if intersection.geom_type=='LineString':
            if intersection.is_empty: # prevents crushing if segment is empty
                print("empty intersection")
                continue

            x, y = intersection.coords.xy

            #line_coords = sorted(line_coords, reverse = True)# i think i do not need this for slope
            #print('sorted:', line_coords)

            x_dif = abs(x[-1] - x[0])
            if x_dif < frame_width*.20: # This should be adjusted now it should skip this frame if a line is less then 20% of the frame width
                #print(i, 'th is too short')
                continue
            else:
                #print(i, "th frame is simple")
                slope, _, _, _, _ = scipy.stats.linregress(x, y)
                #write_run_info("slope:{}".format(slope))
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
                if x_dif < frame_width*.20: # This can be adjusted now it should skip this frame is line is less then 20% of the frame width
                    #print(i, 'th is too short')
                    continue
                else:
                    #print(i, "th frame is complex")
                    slope, _, _, _, _ = scipy.stats.linregress(x, y)
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
                if mean_slopes > 0 and mean_slopes < 2:
                    PlusMinus = 1
                elif mean_slopes < 0 and mean_slopes > -2:
                    PlusMinus = 0
                else:
                    PlusMinus = []
                PlusMinus_index.append([PlusMinus,cut_point])
        else:
            continue

    # Find the middle by the change in a slope of the lines in PlusMinus_index
    cutting_point_detected = 0
    test_seq1 = [0,0,1,1]
    test_seq2 = [1,1,0,0]
    PlusMinus = [x for x,_ in  PlusMinus_index]
    for i in range(len(PlusMinus_index)):
        pm_seq = PlusMinus[i:i+len(test_seq1)]
        if pm_seq != test_seq1 and pm_seq != test_seq2:
            continue
        if cutting_point_detected == 1:
            print('Several cutting points identified, needs to be investigated!')
            write_run_info('Several cutting points identified, needs to be investigated!')
            break
        cutting_point = PlusMinus_index[i+1][1] + ((PlusMinus_index[i+2][1] - PlusMinus_index[i+1][1])/2)
        cutting_point_detected = 1
        #if cutting_point is immediately at the beggining of the sample ignore it
        if cutting_point < imgheight*2: # if cutting point is within 2*image height it will be ignored
            cutting_point_detected = 0

    # Split sequence where it is crossing the middle
    if cutting_point_detected==1:

        print('final_cutting_point:', cutting_point)
        write_run_info('Core sample crosses the center and is cut at: ' + str(cutting_point))
        cut_frame1_poly = shapely.geometry.box(0, 0, cutting_point, imgheight)
        Multi_centerlines1= Multi_centerlines.intersection(cut_frame1_poly)
        cut_frame2_poly = shapely.geometry.box(cutting_point, 0, imgwidth, imgheight)
        Multi_centerlines2= Multi_centerlines.intersection(cut_frame2_poly)
        #write_run_info("Multi_centerlines2 type {}".format(Multi_centerlines2.geom_type))
        measure_points1 = []
        measure_points2 = [] # I initiate it alredy here so i can use it in the test later
        # Reorder Multi_centerlines1
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
            # Order contours by x_maxs
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
            # Find nearest_points for each pair of lines
            for i in range(len(Multi_centerlines2.geoms)-1):
                points = shapely.ops.nearest_points(Multi_centerlines2.geoms[i], Multi_centerlines2.geoms[i+1])
                measure_points2.append(points)

        if not measure_points2:
            measure_points=measure_points1
            Multi_centerlines = Multi_centerlines1
        else:
            measure_points=[measure_points1, measure_points2]
            Multi_centerlines = [Multi_centerlines1, Multi_centerlines2]

        return Multi_centerlines, measure_points, cutting_point

    else:
        # Loop through them to measure pairwise distances between nearest points
        print('middle point was not detected')
        write_run_info('Middle point was not detected')
        cutting_point = {}
        # Reorder the lines
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
        # Find nearest_points for each pair of lines
        measure_points = []
        for i in range(len(Multi_centerlines.geoms)-1):
            points = shapely.ops.nearest_points(Multi_centerlines.geoms[i], Multi_centerlines.geoms[i+1])
            measure_points.append(points)

        return Multi_centerlines, measure_points, cutting_point

#######################################################################
# Plot predicted lines and points of measurements to visually assess
#######################################################################
def plot_lines(image, centerlines, measure_points, file_name, path_out):
    # Create pngs folder in output path
    write_run_info("Plotting output as png")
    print("Plotting output as png")
    export_path = os.path.join(path_out, 'pngs')
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    f = file_name + '.png'

    # Save images at original size unles they are bigger then 30000. Should improve diagnostics on the images
    imgheight, imgwidth = image.shape[:2]
    #print('imgheight, imgwidth', imgheight, imgwidth)
    plot_dpi = 100

    if imgwidth < 30000:
        plt.figure(figsize = (imgwidth/plot_dpi, 2*(imgheight/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
        linewidth = (imgheight/1000)*5   # looks very variable depending on the image resolution whne set as a constant defaoult is 1.5
    else: # adjust image size if it`s exceeding 30000 pixels to 30000
        resized_height = imgheight*(30000/imgwidth)
        plt.figure(figsize = (30000/plot_dpi, 2*(resized_height/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
        linewidth = (resized_height/1000)*5  # looks very variable depending on the image resolution whne set as a constant defaoult is 1.5


    # Plot the lines to the image
    if not isinstance(centerlines, list) and len(centerlines)>2:

        #plt.figure(figsize = (30,15))
        #plt.imshow(image)
        for i in range(len(centerlines.geoms)-1):
            #print('loop', i)
            points = measure_points[i]

            xc,yc = centerlines[i].coords.xy
            plt.plot(xc,yc,'g', linewidth=linewidth)

            xp, yp = points[0].coords.xy
            xp1, yp1 = points[1].coords.xy
            plt.plot([xp, xp1], [yp, yp1], 'r', linewidth=linewidth)

        xc,yc = centerlines[-1].coords.xy
        plt.plot(xc,yc,'g', linewidth=linewidth)
        #plt.show()


    elif isinstance(centerlines, list) and len(centerlines)==2:
        for l in range(2):
            color = ['g', 'b']
            centerlines1 = centerlines[l]
            #print('measure_points:', len(measure_points))
            measure_points1 = measure_points[l]
            if len(measure_points1)==0: # Precaution in case the first part of measure points is empty
                continue
            for i in range(len(centerlines1.geoms)-1):
                #print('loop', i)

                xc,yc = centerlines1[i].coords.xy
                plt.plot(xc,yc,color[l], linewidth=linewidth)

                points = measure_points1[i]
                xp, yp = points[0].coords.xy
                xp1, yp1 = points[1].coords.xy
                plt.plot([xp, xp1], [yp, yp1], 'r', linewidth=linewidth)

            xc,yc = centerlines1[-1].coords.xy # To print the last point
            plt.plot(xc,yc, color[l], linewidth=linewidth)
        #plt.show()

    plt.savefig(os.path.join(export_path, f), bbox_inches = 'tight', pad_inches = 0)

#######################################################################
# Create a JSON file for shiny app
#######################################################################
def write_to_json(image_name, cutting_point, run_ID, path_out, centerlines_rings,
                    clean_contours_rings, clean_contours_cracks=None):
    print("Writing .json file")
    write_run_info("Writing .json file")
    # Define the structure of json
    out_json = {}
    out_json = {image_name: {'run_ID':run_ID, 'predictions':{}, 'directionality': {},
                            'center': {}, 'est_rings_to_pith': {}, 'ring_widths': {}}}
    out_json[image_name]['predictions'] = {'ring_line': {}, 'ring_polygon': {},
                                            'crack_polygon': {}, 'resin_polygon': {},
                                            'pith_polygon': {}}
    out_json[image_name]['center'] = {'cutting_point': cutting_point, 'pith_present': {},
                                        'pith_inferred': {'coords': {'x': [], 'y': []}}}
    out_json[image_name]['ring_widths'] = {'directionality': {}, 'shortest_distance': {},
                                            'manual': {}}
    # Separate x and y coordinates for polygons and line
    if clean_contours_cracks==None:
        input_vars = [centerlines_rings, clean_contours_rings]
    else:
        input_vars = [centerlines_rings, clean_contours_rings, clean_contours_cracks]
    json_names = ['ring_line', 'ring_polygon', 'crack_polygon']
    for v in range(len(input_vars)):
        coords = {}
        for i in range(len(input_vars[v])):

            # 'If else' becasue ring_line is shapely object and clean contours are from opencv and have different structure
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

    output = os.path.join(path_out, os.path.splitext(image_name)[0] + '.json')
    with open(output,'w') as outfile:
        json.dump(out_json, outfile, indent=4)
#######################################################################
# Create a POS file with measure points
#######################################################################
def write_to_pos(centerlines, measure_points, file_name, image_name, DPI, path_out):
    print("Writing .pos file")
    write_run_info("Writing .pos file")
    # Check if it is one or two parts in measure points
    # If two adjust naming. Nothing for the normal one and add "x" at the end for the second part
    #print('measure_point len', len(measure_points))
    #print('measure_point', measure_points)
    # Prepare date, time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    # Prepare unit conversion
    pixel_per_mm = DPI/25.4
    # Create paths for output files
    pos_name = file_name + '.pos'
    posX_name = file_name + 'X' + '.pos'
    out_file_path = os.path.join(path_out, pos_name)
    out_fileX_path = os.path.join(path_out, posX_name)

    #print('names done')
    #print('centerline', centerlines)
    #print('len centrlines', type(centerlines))

    if not isinstance(centerlines, list) and len(centerlines)>2:
        # Prepare points
        str_measure_points = []
        first_x, first_y = measure_points[0][0].coords.xy
        #print('first_xy', first_x, first_y)
        first_point = str(round(float(first_x[0])/pixel_per_mm, 3))+","+str(round(float(first_y[0])/pixel_per_mm, 3))
        str_measure_points.append(first_point)

        for i in range(len(measure_points)-1):
            # This gets second point of a current tuple and the first of the next tuple
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

        # Write in the file
        with open(out_file_path, 'w') as f:
            print('#DENDRO (Cybis Dendro program compatible format) Coordinate file written as', file=f)
            print('#Imagefile {}'.format(image_name), file=f)
            print('#DPI {}'.format(DPI), file=f)
            print('#All coordinates in millimeters (mm)', file=f)
            print('SCALE 1', file=f)
            print('#C DATED', file=f) #before '#C DATED 2018'
            print('#C Written={};'.format(dt_string), file=f)
            print('#C CooRecorder=;', file=f) #before '#C CooRecorder=9.4 Sept 10 2019;'
            print('#C licensedTo=;', file=f) #before '#C licensedTo=Alexis Arizpe, alexis.arizpe@gmi.oeaw.ac.at;'
            for i in str_measure_points:
                print(i, file=f)

    elif isinstance(centerlines, list) and len(centerlines)==2:
        for l in range(2):

            measure_points1 = measure_points[l]
            print('len of measure_points1', len(measure_points1))
            if len(measure_points1)==0: # Precaution in case the first part of measure points is empty
                write_run_info('Middle of the core identified on the first ring!!!Only X .pos file will be created!!!')
                continue
            str_measure_points1 = []
            first_x, first_y = measure_points1[0][0].coords.xy
            #print('first_xy', first_x, first_y)
            first_point = str(round(float(first_x[0])/pixel_per_mm, 3))+","+str(round(float(first_y[0])/pixel_per_mm, 3))
            str_measure_points1.append(first_point)

            for i in range(len(measure_points1)-1):
                # This gets second point of a current tuple and the first of the next tuple
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

            # Write in the file
            out_file_paths = [out_file_path, out_fileX_path]
            with open(out_file_paths[l], 'w') as f:
                print('#DENDRO (Cybis Dendro program compatible format) Coordinate file written as', file=f)
                print('#Imagefile {}'.format(image_name), file=f)
                print('#DPI {}'.format(DPI), file=f)
                print('#All coordinates in millimeters (mm)', file=f)
                print('SCALE 1', file=f)
                print('#C DATED', file=f) #before '#C DATED 2018'
                print('#C Written={};'.format(dt_string), file=f)
                print('#C CooRecorder=;', file=f) #before '#C CooRecorder=9.4 Sept 10 2019;'
                print('#C licensedTo=;', file=f) #before '#C licensedTo=Alexis Arizpe, alexis.arizpe@gmi.oeaw.ac.at;'
                for i in str_measure_points1:
                    print(i, file=f)


#######################################################################
# Run detection on the forlder or images
#######################################################################
def main():
    # get the arguments
    args = get_args()

    # PREPARE THE MODEL
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Prepare ring model
    configRing = TreeRing_onlyRing.TreeRingConfig()

    class InferenceConfig(configRing.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    configRing = InferenceConfig()
    configRing.display()
    # Create model in inference mode
    modelRing = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configRing)
    # Load weights
    weights_path_Ring = args.weightRing
    print("Loading ring weights ")
    modelRing.load_weights(weights_path_Ring, by_name=True)

    # Load cracks model only if cracks weights are provided
    if args.weightCrack is not None:
        # Prepare crack model
        configCrack = TreeRing_onlyCracks.TreeRingConfig()

        class InferenceConfig(configCrack.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        configCrack = InferenceConfig()
        configCrack.display()
        # Create model in inference mode
        modelCrack = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=configCrack)
        # Load weights
        weights_path_Crack = args.weightCrack

        print("Loading crack weights ")
        modelCrack.load_weights(weights_path_Crack, by_name=True)
    else:
        modelCrack = None
    # Define class names
    class_names = ['BG', 'ring']

    # RETRAINING
    if args.dataset is not None:
        print("Starting retraining mode")
        # Check compulsary argument and print which are missing
        if args.weightRing==None:
            print("Compulsory argument --weightRing is missing. Specify the path to ring weight file")
            exit()
        # Check and prepare annotations
        prepareAnnotations(dataset=args.dataset, overwrite_existing=False)

        # Start retraining
        retraining(weights=args.weightRing, dataset=args.dataset, logs=args.logs, start_new=args.start_new)

    # DETECTION
    else:
        print("Starting inference mode")
        # Check compulsory argument and print which are missing
        print('Checking compulsory arguments')
        if args.input==None:
            print("Compulsory argument --input is missing. Specify the path to image file of folder")
            exit()
        if args.weightRing==None:
            print("Compulsory argument --weightRing is missing. Specify the path to ring weight file")
            exit()
        if args.output_folder==None:
            print("Compulsory argument --output_folder is missing. Specify the path to output folder")
            exit()
        if args.dpi==None:
            print("Compulsory argument --dpi is missing. Specify the DPI value for the image")
            exit()
        if args.run_ID==None:
            print("Compulsory argument --run_ID is missing. Specify the Run ID")
            exit()

        path_out = os.path.join(args.output_folder, args.run_ID)
        # Check if output dir for run_ID exists and if not create it
        if not os.path.isdir(path_out):
            os.mkdir(path_out)

        now = datetime.now()
        dt_string_name = now.strftime("D%Y%m%d_%H%M%S") #"%Y-%m-%d_%H:%M:%S"
        dt_string = now.strftime("%Y-%m-%d_%H:%M:%S")
        run_ID = args.run_ID
        log_file_name = str(args.logfile) + run_ID + '_' + dt_string_name + '.log' #"RunID" + dt_string +
        log_file_path =os.path.join(path_out, log_file_name)

        # Initiate log file
        with open(log_file_path,"x") as fi:
            print("Run started:" + dt_string, file=fi)
            print("Ring weights used:" + weights_path_Ring, file=fi)

            if args.weightCrack is not None:
                print("Crack weights used:" + weights_path_Crack, file=fi)

        # Create a list of already exported jsons to prevent re-running the same image
        json_list = []
        for f in os.listdir(path_out):
            if f.endswith('.json'):
                #json_name = os.path.splitext(f)[0]
                json_name = f.replace('.json', '')
                json_list.append(json_name)

        input = args.input
        # Check pathin if its folder or file and get file list of either
        if os.path.isfile(input):
            # Get file name and dir to file
            input_list = [os.path.basename(input)]
            input_path = os.path.split(input)[0]
        elif os.path.isdir(input):
            # Get a list of files in the dir
            input_list = os.listdir(input)
            input_path = input
        else:
            print("Image argument is neither valid file nor directory") # input or image?
            write_run_info("Image argument is neither valid file nor directory")
        #print("got until here", input_list, input_path)

        for f in input_list:
            supported_extensions = ['.tif', '.tiff']
            file_extension = os.path.splitext(f)[1]

            if file_extension in supported_extensions and os.path.splitext(f)[0] in json_list:
                # print image name first to keep the output consistent
                print("Processing image: {}".format(f))
                write_run_info("Processing image: {}".format(f))
                print("JSON FILE FOR THIS IMAGE ALREADY EXISTS IN OUTPUT")
                write_run_info("JSON FILE FOR THIS IMAGE ALREADY EXISTS IN OUTPUT")
            elif file_extension in supported_extensions and os.path.splitext(f)[0] not in json_list:
                try:
                    image_start_time = time.time()
                    print("Processing image: {}".format(f))
                    write_run_info("Processing image: {}".format(f))
                    image_path = os.path.join(input_path, f)
                    im_origin = skimage.io.imread(image_path)

                    # check number of channels and if 4 assume rgba and convert to rgb
                    # conversion if image is not 8bit convert to 8 bit
                    if im_origin.dtype == 'uint8' and im_origin.shape[2] == 3:
                        print("Image was 8bit and RGB")
                        write_run_info("Image was 8bit and RGB")
                    elif im_origin.shape[2] == 4:
                        print("Image has 4 channels, assuming RGBA, trying to convert")
                        write_run_info("Image has 4 channels, assuming RGBA, trying to convert")
                        im_origin = img_as_ubyte(skimage.color.rgba2rgb(im_origin))
                    elif im_origin.dtype != 'uint8':
                        print("Image converted to 8bit")
                        write_run_info("Image converted to 8bit")
                        im_origin = img_as_ubyte(exposure.rescale_intensity(im_origin))  # with rescaling should be better

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
                        detection_rows=int(args.n_detection_rows)

                    detected_mask = sliding_window_detection_multirow(image = im_origin,
                                                            detection_rows=detection_rows,
                                                            modelRing=modelRing,
                                                            modelCrack=modelCrack,
                                                            overlap = sliding_window_overlap,
                                                            cropUpandDown = cropUpandDown)

                    write_run_info("sliding_window_detection_multirow done")
                    print("sliding_window_detection_multirow done")
                    detected_mask_rings = detected_mask[:,:,0]
                    #print("detected_mask_rings", detected_mask_rings.shape)

                    # Define minimum mask overlap if not provided
                    if args.min_mask_overlap is not None:
                        min_mask_overlap = int(args.min_mask_overlap)
                    else:
                        min_mask_overlap = 3

                    clean_contours_rings = clean_up_mask(detected_mask_rings, min_mask_overlap=min_mask_overlap, is_ring=True)
                    write_run_info("clean_up_mask done")
                    print("clean_up_mask done")
                    #print(clean_contours_rings.shape)
                    centerlines_rings = find_centerlines(clean_contours_rings, cut_off=0.01, y_length_threshold=im_origin.shape[0]*0.05)
                    if centerlines_rings is None:
                        write_run_info("IMAGE WAS NOT FINISHED")
                        print("IMAGE WAS NOT FINISHED")
                        continue

                    write_run_info("find_centerlines done")
                    print("find_centerlines done")
                    centerlines, measure_points, cutting_point = measure_contours(centerlines_rings, detected_mask_rings)
                    write_run_info("measure_contours done")
                    print("measure_contours done")
                    # If cracks are detected
                    clean_contours_cracks = None
                    if args.weightCrack is not None:
                        detected_mask_cracks = detected_mask[:,:,1]
                        print("detected_mask_cracks", detected_mask_cracks.shape)
                        clean_contours_cracks = clean_up_mask(detected_mask_cracks, is_ring=False)
                        write_run_info("clean_up_mask cracks done")
                        print("clean_up_mask cracks done")

                    write_to_json(image_name=f, cutting_point=cutting_point, run_ID=run_ID,
                                    path_out=path_out, centerlines_rings=centerlines_rings,
                                    clean_contours_rings=clean_contours_rings,
                                    clean_contours_cracks=clean_contours_cracks)

                    image_name = os.path.splitext(f)[0]
                    DPI = float(args.dpi)
                    write_to_pos(centerlines, measure_points, image_name, f, DPI, path_out)

                    if args.print_detections == "yes":
                        # Ploting lines is moslty for debugging
                        masked_image = im_origin.astype(np.uint32).copy()
                        masked_image = apply_mask(masked_image, detected_mask_rings, alpha=0.2)

                        if args.weightCrack is not None:
                            masked_image = apply_mask(masked_image, detected_mask_cracks, alpha=0.3)

                        plot_lines(masked_image, centerlines, measure_points,
                                    image_name, path_out)
                    write_run_info("IMAGE FINISHED")
                    print("IMAGE FINISHED")
                    image_finished_time = time.time()
                    image_run_time = image_finished_time - image_start_time
                    write_run_info(f"Image run time: {image_run_time} s")

                except Exception as e:
                    write_run_info(e)
                    write_run_info("IMAGE WAS NOT FINISHED")
                    print(e)
                    print("IMAGE WAS NOT FINISHED")


if __name__ == '__main__':
    main()
