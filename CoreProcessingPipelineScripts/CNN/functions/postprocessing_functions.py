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
import skimage
import copy
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel (level = 'warning')
import shapely
from shapely.ops import nearest_points
import scipy
from datetime import datetime
from operator import itemgetter
import logging
import pickle

# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
#print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from functions.src_get_centerline import get_centerline

# set up logger
logger = logging.getLogger(__name__)
#######################################################################
# apply mask to an original image
#######################################################################
def apply_mask(image, mask, alpha=0.5):
    """Apply the given mask to the image. In BGR
    """
    color = (0.0, 0.0, 0.7) # B
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    color = (0.0, 0.7, 0.0) # G
    for c in range(3):
        image[:, :, c] = np.where(mask == 2,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])

    color = (0.7, 0.0, 0.0) # R
    for c in range(3):
        image[:, :, c] = np.where(mask > 2,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

############################################################################################################
# Converts yolov8 result into binary mask
############################################################################################################
def convert_to_binary_mask(result, class_number):
    # result is yolov8 result for one image
    # it will output a binary mask of all the detected masks of desired class_number
    logging.info("convert_to_binary_mask started")
    im_shape = result.orig_shape
    print('im_shape', im_shape)
    cls_list = result.boxes.cls.int().tolist()
    print("cld_list", cls_list)
    if len(cls_list)==0 or class_number not in cls_list:
        binary_mask = np.zeros(im_shape)

    else:
        cld_list_bool = [i == class_number for i in cls_list]
        result_sub = result[cld_list_bool]
        mask_coords = result_sub.masks.xy
        array_length_list = [len(i) for i in mask_coords] # to check for empty arrays
        logging.info(f"array_length_list: {array_length_list}")
        all_mask_coords = [i.astype(np.int32) for i in mask_coords if len(i)>0] # empty array caused segmentation fault in cv2.fillPoly
        # convert coords to binary mask of an image
        mask = np.zeros(im_shape)
        logging.info("cv2.fillPoly starts")
        binary_mask = cv2.fillPoly(mask, pts=all_mask_coords, color=1)
        logging.info("cv2.fillPoly finished")

    logging.info("convert_to_binary_mask finished")
    return binary_mask

############################################################################################################
# Sliding window detection with rotation of each part of image by 90 and 45 degrees and combining the output
############################################################################################################
def sliding_window_detection_multirow(image, detection_rows=1, model=None, cracks=False, overlap=0.75, row_overlap=0.1, cropUpandDown=0.17, px_to_crop = 10):
    # The mask for ring is in position the_mask_clean_origin_size[:,:,0] while cracks in the_mask_clean_origin_size[:,:,1]
    # px_to_crop - how many pixels on the edges of detected mask to replace with zeros to clean the edges
    print("sliding_window_detection_multirow started")
    logger.info("sliding_window_detection_multirow started")
    logger.info(f"Sliding window overlap = {overlap} and cropUpandDown = {cropUpandDown}")
    # Crop image top and bottom to avoid detectectig useles part of the image
    imgheight_origin, imgwidth_origin = image.shape[:2]

    #print('image shape', image.shape[:2])
    #print('cropUpandDown', cropUpandDown)
    to_crop = int(imgheight_origin*cropUpandDown)
    new_image = image[to_crop:(imgheight_origin-to_crop), :, :]
    #print('new image shape', new_image.shape)

    imgheight_for_pad, imgwidth_for_pad = new_image.shape[:2]

    # add zero padding at the begining and the end according to overlap
    ## if overlap >= 0.5, every part of the picture is detected same number of time
    if overlap >= 0.5 and overlap < 1:
        front_pad_width = int(imgheight_for_pad*overlap)
        back_pad_width = imgheight_for_pad
    ## if overlap < 0.5, do not need front pad as it will ot overlap equaly anyways
    elif overlap < 0.5:
        front_pad_width = 0
        back_pad_width = int(imgheight_for_pad - (imgheight_for_pad*overlap))
    ## if overlap == 0 no need to pad at all
    elif overlap == 0:
        front_pad_width = 0
        back_pad_width = 0
    else:
        logging.info(f"sliding_window_overlap value of {overlap} is not valid, it should be between 0 and smaller than 1")
        raise SystemExit(f"sliding_window_overlap value of {overlap} is not valid, it should be between 0 and smaller than 1")


    zero_padding_front = np.zeros(shape=(imgheight_for_pad, front_pad_width, 3), dtype='uint8')
    zero_padding_back = np.zeros(shape=(imgheight_for_pad, back_pad_width, 3), dtype='uint8')
    im_padded = np.concatenate((zero_padding_front, new_image, zero_padding_back), axis=1)
    logger.info(f"im_padded.dtype: {im_padded.dtype}") # should be uint8

    imgheight, imgwidth = im_padded.shape[:2]
    logger.info(f'im_padded.shape: {im_padded.shape}')

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
    looping_range = range(0, imgwidth, int(row_height-(row_height*overlap)))
    looping_list = [i for i in looping_range if i < imgwidth-row_height] # before the condition was int(row_height-(row_height*overlap)) + imgwidth_origin
    logger.info(f'looping_list: {looping_list}')

    if cracks is True:
        classes = [0, 1]
    else:
        classes = [0]

    combined_masks_per_class = np.zeros(shape=(imgheight, imgwidth, len(classes)), dtype=int) # combine all the partial masks in the final size of full tiff
    logger.info(f'combined_masks_per_class.shape: {combined_masks_per_class.shape}')
    for rl in row_looping_range:
        logger.info(f"rl: {rl}")
        for i in looping_list: # defines the slide value
            logger.info(f"i: {i}")
            # crop the image
            cropped_part = im_padded[rl:rl+row_height, i:i+row_height]
            print('cropped_part, i, i+imheight', cropped_part.shape, i, i+imgheight)
            logger.info(f'cropped_part, i, i+imheight: {cropped_part.shape}, {i}, {i+imgheight}')
            #logger.info('cropped_part.dtype', cropped_part.dtype) # should be uint8

            # Run detection on the cropped part of the image
            ## Prepare the rotated images 90, 45
            cropped_part_90 = skimage.transform.rotate(cropped_part, angle=90, preserve_range=True).astype(np.uint8)
            cropped_part_45 = skimage.transform.rotate(cropped_part, angle=45,
                                                       preserve_range=True, resize=True).astype(np.uint8)

            ## Run the detection on all 3 at the same time
            logger.info('Detection starts')
            results = model([cropped_part, cropped_part_90, cropped_part_45])
            logger.info('Detection finished')

            for class_number in classes:
                ## create flattened binary masks for every detected image and given class
                r_mask = convert_to_binary_mask(results[0], class_number) # 0 degree mask
                r1_mask = convert_to_binary_mask(results[1], class_number) # 90 degree mask
                r2_mask = convert_to_binary_mask(results[2], class_number) # 45 degree mask

                ## Rotate maskr1 masks back
                r1_mask_back = np.rot90(r1_mask, k=-1)

                # Rotate back and crop to the right size. Beware different dimensions!!!
                imheight2 = r2_mask.shape[0]
                r2_mask_back = skimage.transform.rotate(r2_mask, angle=-45, resize=False)
                to_crop = int((imheight2 - row_height)/2)
                r2_mask_back_cropped = r2_mask_back[to_crop:(to_crop+int(row_height)), to_crop:(to_crop+int(row_height))]

                ## Put all togather
                logger.info(f"r_mask: {r_mask.shape}")
                logger.info(f"r1_mask_back: {r1_mask_back.shape}")
                logger.info(f"r2_mask_back_cropped: {r2_mask_back_cropped.shape}")
                combined_mask_section = r_mask + r1_mask_back + r2_mask_back_cropped
                logger.info(f"combined_mask_section.shape{combined_mask_section.shape}")
                # Crop the edges of detected square to get cleaner mask
                section_cleaned_edges = np.zeros(shape=combined_mask_section.shape, dtype='uint8')
                section_cleaned_edges[px_to_crop:-px_to_crop, px_to_crop:-px_to_crop] = combined_mask_section[px_to_crop:-px_to_crop, px_to_crop:-px_to_crop]

                logger.info(f"section_cleaned_edges.shape{section_cleaned_edges.shape}")
                combined_masks_per_class[rl:rl+row_height, i:i+row_height, class_number] = combined_masks_per_class[rl:rl+row_height, i:i+row_height, class_number] + section_cleaned_edges

    # First remove the padding
    pad_front = zero_padding_front.shape[1]
    logger.info(f'pad_front: {pad_front}')
    pad_back = zero_padding_back.shape[1]
    the_mask_clean = combined_masks_per_class[:,pad_front:-pad_back,:]
    logger.info(f'the_mask_clean.shape: {the_mask_clean.shape}')

    # Concatanete the top and buttom to fit the original image
    missing_part = int((imgheight_origin - the_mask_clean.shape[0])/2)
    to_concatenate = np.zeros(shape=(missing_part, imgwidth_origin, the_mask_clean.shape[2]), dtype='uint8')
    #print("to_concatenate", to_concatenate.shape)
    the_mask_clean_origin_size = np.concatenate((to_concatenate, the_mask_clean, to_concatenate),axis=0)
    logger.info(f'the_mask_clean_origin_size: {the_mask_clean_origin_size.shape}')

    return the_mask_clean_origin_size

#######################################################################
# Extract distances from the mask
#######################################################################
def clean_up_mask(mask, min_mask_overlap=3, is_ring=True):
    # Detects countours of the masks, removes small contours
    print("clean_up_mask started")
    logger.info("clean_up_mask started")
    # Make the mask binary
    binary_mask = np.where(mask >= min_mask_overlap, 255, 0) # this part can be cleaned to remove some missdetections setting condition for higher value
    #print("binary_mask shape", binary_mask.shape)
    #plt.show()
    #type(binary_mask)
    uint8binary = binary_mask.astype(np.uint8).copy()

    # Extract contour coordinates from binary mask
    contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print('contour_shape:', len(contours))
    logger.info(f'Raw_contours: {len(contours)}')

    # Here i extract dimensions and angle of individual contours bigger than threshold
    imgheight, imgwidth = mask.shape[:2]
    if is_ring==True:
        min_size_threshold = imgheight/5 #imgheight/12 # Will take only contours that are bigger than 1/5 of the image
        logger.info(f'min_size_threshold for ring: {min_size_threshold}')
    else:
        min_size_threshold = 1
    contours_filtered = []
    x_mins = []
    for i in range(0, len(contours)):
        x_only = []
        for p in range(len(contours[i])):
            [[x,_]] = contours[i][p]
            x_only.append(x)
        x_min = np.min(x_only)
        #remove those that are too short
        rect = cv2.minAreaRect(contours[i])
        #print(rect)
        imgheight, imgwidth = mask.shape[:2]
        #print(imgheight)
        dim1, dim2 = rect[1]
        dim_max = max([dim1, dim2])
        if dim_max > min_size_threshold:
            contours_filtered.append(contours[i])
            x_mins.append(x_min)
            print("contour shape", contours[i].shape)

    print('Filtered_contours:', len(contours_filtered))
    logger.info(f'Filtered_contours: {len(contours_filtered)}')

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
        #with open(f'shapely_polygon{i}.pkl', 'wb') as file:
        #    pickle.dump(polygon, file)
        try:
            #cline = get_centerline(polygon)
            cline = get_centerline(polygon, segmentize_maxlen=0.5, max_points=600, simplification=0.1,
                                   segmentize_maxlen_post=11, smooth_sigma=5)
        except Exception as e:
            logger.info('Centerline of the ring {} failed with exception {}'.format(i, e))
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
            logger.info(f'Contour {i} was skipped because with line_y_diff {line_y_diff} was less then '
                        f'threshold y_length_threshold {y_length_threshold}')
            continue
        else:
            centerlines.append(cline)

    # test if centerline list contains something and if not abort and give a message
    if not centerlines: # empty list is False
        print("NO LINES LEFT AFTER CLEANING")
        logger.info("NO LINES LEFT AFTER CLEANING")
        logger.info("One reason could be that your images have too much background."
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
    logger.info("Image has height {} and width {}".format(imgheight, imgwidth))
    logger.info("{} ring boundries were detected".format(len(Multi_centerlines.geoms)))

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
                logger.info("empty intersection")
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
                #logger.info("slope:{}".format(slope))
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
                    #logger.info("slope:{}".format(slope))

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
            logger.info('Several cutting points identified, needs to be investigated!')
            break
        cutting_point = PlusMinus_index[i+1][1] + ((PlusMinus_index[i+2][1] - PlusMinus_index[i+1][1])/2)
        cutting_point_detected = 1
        #if cutting_point is immediately at the beggining of the sample ignore it
        if cutting_point < imgheight*2: # if cutting point is within 2*image height it will be ignored
            cutting_point_detected = 0

    # Split sequence where it is crossing the middle
    if cutting_point_detected==1:

        print('final_cutting_point:', cutting_point)
        logger.info('Core sample crosses the center and is cut at: ' + str(cutting_point))
        cut_frame1_poly = shapely.geometry.box(0, 0, cutting_point, imgheight)
        Multi_centerlines1= Multi_centerlines.intersection(cut_frame1_poly)
        cut_frame2_poly = shapely.geometry.box(cutting_point, 0, imgwidth, imgheight)
        Multi_centerlines2= Multi_centerlines.intersection(cut_frame2_poly)
        #logger.info("Multi_centerlines2 type {}".format(Multi_centerlines2.geom_type))
        measure_points1 = []
        measure_points2 = [] # I initiate it alredy here so i can use it in the test later
        # Reorder Multi_centerlines1
        x_maxs = []
        x_mins = []
        for i in range(len(Multi_centerlines1.geoms)):
            minx, _, maxx, _ = Multi_centerlines1.geoms[i].bounds
            x_maxs.append(maxx)
            x_mins.append(minx)

        x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins))/2
        #print('x_middle, x_maxs, x_mins', x_middle, x_maxs, x_mins)
        contourszip = zip(x_middle, Multi_centerlines1.geoms)

        #print('contourszip', contourszip)
        #print('x_maxs', x_maxs)
        centerlines1 = [x for _, x in sorted(contourszip, key=itemgetter(0))]
        Multi_centerlines1 = shapely.geometry.MultiLineString(centerlines1)
        #print('ordered centerlines2:', Multi_centerlines2.geom_type)
        for i in range(len(Multi_centerlines1.geoms)-1):
            points = shapely.ops.nearest_points(Multi_centerlines1.geoms[i], Multi_centerlines1.geoms[i+1])
            measure_points1.append(points)

        if Multi_centerlines2.geom_type=='LineString':
            print("Multi_centerlines2 is only one line")
            logger.info("Multi_centerlines2, the part after cutting point, is only one line")
        else:
            # Order contours by x_maxs
            x_maxs = []
            x_mins = []
            for i in range(len(Multi_centerlines2.geoms)):
                minx, _, maxx,_ = Multi_centerlines2.geoms[i].bounds
                x_maxs.append(maxx)
                x_mins.append(minx)

            x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins))/2
            contourszip = zip(x_middle, Multi_centerlines2.geoms)

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
        logger.info('Middle point was not detected')
        cutting_point = {}
        # Reorder the lines
        x_maxs = []
        x_mins = []
        for i in range(len(Multi_centerlines.geoms)):
            minx, _, maxx,_ = Multi_centerlines.geoms[i].bounds
            x_maxs.append(maxx)
            x_mins.append(minx)

        x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins))/2
        contourszip = zip(x_middle, Multi_centerlines.geoms)

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
# Plot contours
#######################################################################
def plot_contours(image, contours, file_name, path_out):
    # ploty image with extracted contours to facilitate debuging
    image_copy = copy.deepcopy(image)
    for contour in contours:
        cv2.drawContours(image_copy, contour, -1, (0, 255, 0), 1)

    logger.info("Plotting output as png")
    print("Plotting output as png")
    export_path = os.path.join(path_out, 'pngs')
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    f = file_name + '.png'
    # Save images at original size unles they are bigger then px in  length 30000. Should improve diagnostics on the images
    imgheight, imgwidth = image_copy.shape[:2]
    # since I use cv2 to load image I need to convert it to RGB before plotting with matplotlib
    print("image.dtype", image.dtype)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    # print('imgheight, imgwidth', imgheight, imgwidth)
    plot_dpi = 100

    if imgwidth < 30000:
        plt.figure(figsize=(imgwidth / plot_dpi, 2 * (imgheight / plot_dpi)), dpi=plot_dpi)
        # fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image_copy)
    else:  # adjust image size if it`s exceeding 30000 pixels to 30000
        resized_height = imgheight * (30000 / imgwidth)
        plt.figure(figsize=(30000 / plot_dpi, 2 * (resized_height / plot_dpi)), dpi=plot_dpi)
        # fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image_copy)
    plt.savefig(os.path.join(export_path, f), bbox_inches='tight', pad_inches=0)
#######################################################################
# Plot predicted lines and points of measurements to visually assess
#######################################################################
def plot_lines(image, centerlines, measure_points, file_name, path_out):
    # Create pngs folder in output path
    logger.info("Plotting output as png")
    print("Plotting output as png")
    export_path = os.path.join(path_out, 'pngs')
    if not os.path.exists(export_path):
        os.makedirs(export_path)

    f = file_name + '.png'
    # Save images at original size unles they are bigger then px in  length 30000. Should improve diagnostics on the images
    imgheight, imgwidth = image.shape[:2]
    # since I use cv2 to load image I need to convert it to RGB before plotting with matplotlib
    print("image.dtype", image.dtype)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print('imgheight, imgwidth', imgheight, imgwidth)
    plot_dpi = 100

    if imgwidth < 30000:
        plt.figure(figsize = (imgwidth/plot_dpi, 2*(imgheight/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
        linewidth = (imgheight/1000)*3   # looks very variable depending on the image resolution whne set as a constant defaoult is 1.5
    else: # adjust image size if it`s exceeding 30000 pixels to 30000
        resized_height = imgheight*(30000/imgwidth)
        plt.figure(figsize = (30000/plot_dpi, 2*(resized_height/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
        linewidth = (resized_height/1000)*3  # looks very variable depending on the image resolution whne set as a constant defaoult is 1.5

    if centerlines:
        if centerlines.geom_type == 'LineString':
            xc, yc = centerlines.coords.xy
            plt.plot(xc, yc, 'g', linewidth=linewidth)

        # Plot the lines to the image
        elif not isinstance(centerlines, list) and len(centerlines.geoms)>2:

            #plt.figure(figsize = (30,15))
            #plt.imshow(image)
            for i in range(len(centerlines.geoms)-1):
                #print('loop', i)

                xc, yc = centerlines.geoms[i].coords.xy
                plt.plot(xc,yc,'g', linewidth=linewidth)

                if measure_points:
                    points = measure_points[i]
                    xp, yp = points[0].coords.xy
                    xp1, yp1 = points[1].coords.xy
                    plt.plot([xp, xp1], [yp, yp1], 'r', linewidth=linewidth)

            xc, yc = centerlines.geoms[-1].coords.xy
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

                    xc,yc = centerlines1.geoms[i].coords.xy
                    plt.plot(xc,yc,color[l], linewidth=linewidth)

                    points = measure_points1[i]
                    xp, yp = points[0].coords.xy
                    xp1, yp1 = points[1].coords.xy
                    plt.plot([xp, xp1], [yp, yp1], 'r', linewidth=linewidth)

                xc,yc = centerlines1.geoms[-1].coords.xy # To print the last point
                plt.plot(xc,yc, color[l], linewidth=linewidth)
            #plt.show()

    plt.savefig(os.path.join(export_path, f), bbox_inches = 'tight', pad_inches = 0)

#######################################################################
# Create a JSON file for shiny app
#######################################################################
def write_to_json(image_name, cutting_point, run_ID, path_out, centerlines_rings,
                    clean_contours_rings, clean_contours_cracks=None):
    print("Writing .json file")
    logger.info("Writing .json file")
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
        # 'If else' becasue ring_line is shapely object and clean contours are from opencv and have different structure
        #print(json_names[v]) #### just for debug, remove
        if json_names[v] == 'ring_line':
            for i in range(len(input_vars[v].geoms)):
                x_list, y_list = input_vars[v].geoms[i].coords.xy
                x_list = x_list.tolist()
                y_list = y_list.tolist()
        else:
            for i in range(len(input_vars[v])):
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
    logger.info("Writing .pos file")
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

    if not isinstance(centerlines, list) and len(centerlines.geoms)>2:
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
                logger.info('Middle of the core identified on the first ring!!!Only X .pos file will be created!!!')
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
