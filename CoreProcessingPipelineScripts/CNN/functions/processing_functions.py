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
import cv2
import ujson
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.set_loglevel (level = 'warning')
import shapely
from shapely import LineString, MultiLineString, GeometryCollection, Polygon, Point
from shapely.ops import nearest_points
import pygeoops
from datetime import datetime
from operator import itemgetter
import logging

# set up logger
logger = logging.getLogger(__name__)
#######################################################################
# logging and printing with one function
########################################################################
def log_and_print(msg, logger, level="info"):
    print(msg)
    getattr(logger, level)(msg)
#######################################################################
# apply mask to an original image
########################################################################
def apply_mask(image, mask, alpha=0.5):

    image = image.astype(np.float32)

    COLORS = np.array([
        [0, 0, 178.5],  # mask==1
        [0, 178.5, 0],  # mask==2
        [178.5, 0, 0]  # mask>2
    ], dtype=np.float32)

    masks = (
        (mask == 1, COLORS[0]),
        (mask == 2, COLORS[1]),
        (mask > 2, COLORS[2]),
    )

    for m, color in masks:
        image[m] = image[m] * (1 - alpha) + color * alpha

    return image.astype(np.uint8)

############################################################################################################
# Converts yolov8 result into binary mask
############################################################################################################
"""
def convert_to_binary_mask(result, class_number):
    # result is yolo result for one image
    # it will output a binary mask of all the detected masks of desired class_number
    logger.debug("convert_to_binary_mask START")
    im_shape = result.orig_shape
    logger.debug(f"im_shape: {im_shape}")
    cls_list = result.boxes.cls.int().tolist()
    logger.debug(f"cld_list: {cls_list}")
    if len(cls_list) == 0 or class_number not in cls_list:
        binary_mask = np.zeros(im_shape)

    else:
        cld_list_bool = [i == class_number for i in cls_list]
        result_sub = result[cld_list_bool]
        mask_coords = result_sub.masks.xy
        #array_length_list = [len(i) for i in mask_coords]  # to check for empty arrays  ## seem redundant and could be removed if not crushing
        logger.debug(f"array_length_list: {[len(i) for i in mask_coords]}")
        all_mask_coords = [i.astype(np.int32) for i in mask_coords if len(i) > 0]  # empty array caused segmentation fault in cv2.fillPoly
        # convert coords to binary mask of an image
        mask = np.zeros(im_shape)
        logger.debug("cv2.fillPoly starts")
        binary_mask = cv2.fillPoly(mask, pts=all_mask_coords, color=1)
        logger.debug("cv2.fillPoly finished")

    logger.debug("convert_to_binary_mask FINISH")
    return binary_mask
"""
def convert_to_binary_mask(result, class_number):
    logger.debug("convert_to_binary_mask START")
    im_shape = result.orig_shape
    cls = result.boxes.cls.int()

    if len(cls) == 0:
        return np.zeros(im_shape, dtype=np.uint8)

    class_mask = cls == class_number

    if not torch.any(class_mask):
        return np.zeros(im_shape, dtype=np.uint8)

    result_sub = result[class_mask]

    all_mask_coords = [
        poly.astype(np.int32)
        for poly in result_sub.masks.xy
        if len(poly)
    ]

    if not all_mask_coords:
        return np.zeros(im_shape, dtype=np.uint8)

    mask = np.zeros(im_shape, dtype=np.uint8)

    cv2.fillPoly(mask, all_mask_coords, 1)
    logger.debug("convert_to_binary_mask FINISH")
    return mask
############################################################################################################
# Sliding window detection with rotation of each part of image by 90 and 45 degrees and combining the output
############################################################################################################
def _rotate_image(image, angle):
    """
    Rotate image by arbitrary angle while preserving all pixels.
    Parameters:
    image : ndarray
    angle : float
    Returns: ndarray
    """

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = abs(M[0, 0]), abs(M[0, 1])

    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)

    M[0, 2] += new_w / 2 - center[0]
    M[1, 2] += new_h / 2 - center[1]

    return cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

def _crop_and_rotate(im_padded, rl, row_height, i):
    cropped_part = im_padded[rl:rl + row_height, i:i + row_height]

    ## Prepare the rotated image 45
    cropped_part_45 = _rotate_image(cropped_part, 45)
    #cropped_part_45 = skimage.transform.rotate(cropped_part, angle=45,
    #                                           preserve_range=True, resize=True).astype(np.uint8)

    return (cropped_part, np.rot90(cropped_part, k=1), cropped_part_45)

def _rotate_masks_back(results, row_height, class_number):
    ## create flattened binary masks for every detected image and given class and sum them up
    #getting r_mask
    combined_mask = np.ascontiguousarray(convert_to_binary_mask(results[0], class_number), dtype='int8')  # 0 degree mask


    ## Rotate maskr1 masks back and add to combined
    combined_mask += np.ascontiguousarray(np.rot90(convert_to_binary_mask(results[1], class_number), k=-1), dtype='int8')

    # Rotate back and crop r2_mask to the right size. Beware different dimensions!!!
    r2_mask_back = _rotate_image(convert_to_binary_mask(results[2], class_number), -45)
    #r2_mask_back = skimage.transform.rotate(convert_to_binary_mask(results[2], class_number), angle=-45, resize=False)
    imheight = r2_mask_back.shape[0]
    to_crop = (imheight - row_height) // 2
    combined_mask += np.ascontiguousarray(r2_mask_back[to_crop:(to_crop + int(row_height)), to_crop:(to_crop + int(row_height))], dtype='int8')

    logger.debug("combined_mask shape: %s", combined_mask.shape)

    return combined_mask

def _concat_top_bottom(to_crop, imgwidth_origin, the_mask_clean):
    to_concatenate = np.zeros(shape=(to_crop, imgwidth_origin, the_mask_clean.shape[2]), dtype='int8')
    return np.concatenate((to_concatenate, the_mask_clean, to_concatenate), axis=0)

def _clean_and_combine(combined_masks_per_class, binary_masks_back, class_number, rl, i, px_to_crop, row_height):
    #combined_mask_section = binary_masks_back[0] + binary_masks_back[1] + binary_masks_back[2]

    logger.debug("combined_mask_section.shape: %s", binary_masks_back.shape)
    # Crop the edges of detected square to get cleaner mask
    #section_cleaned_edges = _numba_clean(combined_mask_section, px_to_crop)
    if px_to_crop == 0:
        section_cleaned_edges = binary_masks_back
    else:
        section_cleaned_edges = np.zeros(shape=binary_masks_back.shape, dtype='int8')
        section_cleaned_edges[px_to_crop:-px_to_crop, px_to_crop:-px_to_crop] = binary_masks_back[
                                                                                px_to_crop:-px_to_crop,
                                                                                px_to_crop:-px_to_crop]

    logger.debug("section_cleaned_edges.shape: %s", section_cleaned_edges.shape)
    combined_masks_per_class[rl:rl + row_height, i:i + row_height, class_number] += section_cleaned_edges

    return combined_masks_per_class

def sliding_window_detection_multirow(image, detection_rows=1, model=None, cracks=False, overlap=0.75, row_overlap=0.1, cropUpandDown=0.17, px_to_crop = 10):
    # The mask for ring is in position the_mask_clean_origin_size[:,:,0] while cracks in the_mask_clean_origin_size[:,:,1]
    # px_to_crop - how many pixels on the edges of detected mask to replace with zeros to clean the edges
    #print("sliding_window_detection_multirow started")
    logger.info("sliding_window_detection_multirow START")
    logger.debug("Sliding window overlap: %s, and cropUpandDown: %s", overlap, cropUpandDown)
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
        logger.warning(f"sliding_window_overlap value of {overlap} is not valid, it should be between 0 and smaller than 1")
        raise SystemExit(f"sliding_window_overlap value of {overlap} is not valid, it should be between 0 and smaller than 1")

    zero_padding_front = np.zeros(shape=(imgheight_for_pad, front_pad_width, 3), dtype='uint8')
    zero_padding_back = np.zeros(shape=(imgheight_for_pad, back_pad_width, 3), dtype='uint8')
    im_padded = np.concatenate((zero_padding_front, new_image, zero_padding_back), axis=1)
    logger.debug("im_padded.dtype: %s", im_padded.dtype)  # should be uint8

    imgheight, imgwidth = im_padded.shape[:2]
    logger.debug("im_padded.shape: %s", im_padded.shape)

    # Define sliding window parameters
    ## rows
    if detection_rows > 1:
        row_overlap = row_overlap
        row_overlap_height = int((imgheight*row_overlap)/(detection_rows-1))
        row_height = int(((row_overlap_height*(detection_rows-1))+imgheight)/detection_rows)
        row_looping_range = range(0, imgheight-(row_height-1), int(row_height-row_overlap_height))
    else:
        row_looping_range = (0,)  # comma makes it tuple and iterable without it its just an integer
        row_height = imgheight

    ## columns
    looping_range = range(0, imgwidth, int(row_height-(row_height*overlap)))
    looping_list = [i for i in looping_range if i < imgwidth-row_height]  # before the condition was int(row_height-(row_height*overlap)) + imgwidth_origin
    logger.debug('looping_list: %s', looping_list)

    classes = (0, 1) if cracks else (0,)

    combined_masks_per_class = np.zeros(shape=(imgheight, imgwidth, len(classes)), dtype='int8') # combine all the partial masks in the final size of full tiff
    logger.debug("combined_masks_per_class.shape: %s", combined_masks_per_class.shape)
    for rl in row_looping_range:
        logger.debug("rl: %s", rl)
        for i in looping_list:  # defines the slide value
            logger.debug("i: %s", i)

            ## Run the detection on all 3 rotations at the same time
            logger.debug("CNN detection starts")
            results = model(_crop_and_rotate(im_padded, rl, row_height, i))
            logger.debug("CNN detection finished")

            for class_number in classes:
                # Rotate masks back
                binary_masks_back = _rotate_masks_back(results, row_height, class_number) #r_mask, r1_mask_back, r2_mask_back_cropped

                ## Put all togather
                combined_masks_per_class = _clean_and_combine(combined_masks_per_class, binary_masks_back, class_number,
                                                              rl, i, px_to_crop, row_height)

    # First remove the padding
    the_mask_clean = combined_masks_per_class[:, front_pad_width:-back_pad_width, :]
    logger.debug("the_mask_clean.shape: %s", the_mask_clean.shape)

    # Concatanete the top and buttom to fit the original image
    the_mask_clean_origin_size = _concat_top_bottom(to_crop, imgwidth_origin, the_mask_clean)
    logger.debug("the_mask_clean_origin_size: %s", the_mask_clean_origin_size.shape)

    logger.info("sliding_window_detection_multirow FINISH")
    return the_mask_clean_origin_size

#######################################################################
# Extract distances from the mask
#######################################################################
def clean_up_mask(mask, min_mask_overlap=3, is_ring=True, simplify_tolerance=0):
    # Detects countours of the masks, removes small contours
    logger.info("clean_up_mask START")
    logger.info(f'is_ring: {is_ring}')
    # Make the mask binary
    uint8binary = (mask >= min_mask_overlap).astype(np.uint8) * 255

    # Extract contour coordinates from binary mask
    contours, _ = cv2.findContours(uint8binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print('contour_shape:', len(contours))
    logger.debug("Raw_contours: %s", len(contours))

    # Here I extract dimensions and angle of individual contours bigger than threshold
    imgheight, imgwidth = mask.shape[:2]
    if is_ring:
        min_size_threshold = imgheight/5 #imgheight/12 # Will take only contours that are bigger than 1/5 of the image height
        logger.debug("min_size_threshold for ring: %s", min_size_threshold)
    else:
        min_size_threshold = 1

    # Filter the size and convert into shapely polygons and simplify
    contours_filtered = []
    x_mins = []
    for contour in contours:
        x_min = contour[:, 0, 0].min()
        logger.debug("contour.shape: %s", contour.shape)
        # Check the number of points as Polygon requires at least 4
        n_points = contour.shape[0]
        #print("n_points", n_points)
        logger.debug("Contour has %s points", n_points)
        # Remove those that are too short
        dim_max = max(cv2.minAreaRect(contour)[1])
        if dim_max > min_size_threshold and n_points >= 4:
            # Convert in shapely polygon
            cont_polygon = shapely.geometry.Polygon(contour[:, 0, :])
            simpl_cont_polygon = cont_polygon.simplify(tolerance=simplify_tolerance, preserve_topology=True)
            # Check if polygons are bigger than 0 and valid if not try to convert
            if simpl_cont_polygon.area > 0:
                logger.debug("Contour area > 0")

                if shapely.is_valid(simpl_cont_polygon) and isinstance(simpl_cont_polygon, Polygon):
                    logger.debug("Geometry is valid and a Polygon")
                    contours_filtered.append(simpl_cont_polygon)
                    x_mins.append(x_min)
                    logger.debug("x_mins: %s", x_mins)

                else:
                    logger.debug("Geometry is not valid or not a Polygon")
                    cont_val = shapely.make_valid(simpl_cont_polygon)
                    # If multiple created get the one with biggest area.
                    # Assuming some miniture selfintersections around pixels at the edges.
                    # Sometimes there are nested multipolygons hance the while loop
                    while not isinstance(cont_val, Polygon):
                        cont_val = max(cont_val.geoms, key=lambda a: a.area)

                    if isinstance(cont_val, Polygon):
                        contours_filtered.append(cont_val)
                        x_mins.append(x_min)
                        logger.debug("x_mins: %s", x_mins)
                    else:
                        logger.warning(f'Contour not appended because it is {cont_val.geom_type} and expect Polygon')

            #print("contour shape", contours[i].shape
    logger.debug("contours_filtered_n: %s", len(contours_filtered))
    # Order contours by x, e.g. from left to right
    contours_out = tuple(contour for _, contour in sorted(zip(x_mins, contours_filtered), key=itemgetter(0)))

    logger.info("clean_up_mask FINISH")
    # Returns filtered and ordered contours in a form of tuple of shapely polygons
    return contours_out

#######################################################################
# Finds centerlines in contours
#######################################################################
def find_centerlines(clean_contours, cut_off=0.01, y_length_threshold=100, simplification_tolerance=0):
    # Find ceneterlines in polygons
    # cut_off clips upper and lower edges which are sometimes turning horizontal and affect measurements
    # y_length_threshold removes lines that are too short on y axes thus most probably horizontal misdetections
    logger.info("find_centerlines START")

    centerlines = []
    for i, polygon in enumerate(clean_contours):
        logger.debug("ring_contour: %s", i)

        try:
            cline = pygeoops.centerline(polygon, densify_distance=-0.1, min_branch_length=-10, simplifytolerance=-0.20, extend=False)
            # min_branch_length=-10 will filter out all branches shorter than 10 times polygon width. In problems when cline is multilinstring its because of branches.
            # the value does not affect performance only if its 0 because its probably skipping section of code
            # simplifytolerance is simplifying the line with 0 no simplification and -0.20 seems to be reasonable. with no simplification the lines are too wigly
            # seems it does not affect performance
            # densify_distance=-1 segmentize the polygon sections longer then one average polygon width. The -1 seems to work well at least for now.
            # higher value means less points and worse line or none (-0.8 was also a good value) current testing -0.1 is the best
            # affects perfomance a lot
            # extent was False from the begining but do not have any notes why
            logger.debug("cline: %s", cline)
            if isinstance(cline, MultiLineString):
                cline = max(cline.geoms, key=lambda a: a.length)
            elif not isinstance(cline, LineString):
                logger.warning(f"cline in neither LineString nor MultilineString: {cline}")
                continue

            centerlines.append(cline)
            # minx, miny, maxx, maxy = cline.bounds
            logger.debug("Cline min x,y: %s, %s and max x,y: %s, %s", *cline.bounds )

        except Exception as e:
            log_and_print(f"Centerline of the ring {i} failed with exception {e}", logger, "warning")
            continue

    # test if centerline list contains something and if not abort and give a message
    if not centerlines: # empty list is False
        log_and_print("NO LINES LEFT AFTER CLEANING", logger, "warning")
        log_and_print("One reason could be that your images have too much background."
                    "Ideally, there should not be too much background above and below the core."
                    "Try to crop tighter.", logger, "warning")
        return

    else:
        logger.info(f'Filtered_centerlines: {len(centerlines)}')
        ## Cut off upper and lower part of detected lines. It should help with problems of horizontal ends of detections
        Multi_centerlines_to_crop = shapely.geometry.MultiLineString(centerlines)
        minx, miny, maxx, maxy = Multi_centerlines_to_crop.bounds
        px_to_cut_off = int((maxy-miny)*cut_off)
        logger.debug("px_to_cut_off: %s", px_to_cut_off)
        logger.debug("minx: %s, miny: %s, maxx: %s, maxy: %s", minx, miny, maxx, maxy)
        frame_to_crop = shapely.geometry.box(minx, miny+px_to_cut_off, maxx, maxy-px_to_cut_off)
        Multi_centerlines_cropped = Multi_centerlines_to_crop.intersection(frame_to_crop)
        # To check if it crops something
        #minx, miny, maxx, maxy = Multi_centerlines.bounds
        #print('minx, miny, maxx, maxy after', minx, miny, maxx, maxy)

        # Remove too short lines based on the threshold and simplify the number of points in order to reduce final size
        if isinstance(Multi_centerlines_cropped, MultiLineString):
            Centerlines_clean = [l.simplify(tolerance=simplification_tolerance, preserve_topology=False) for l in Multi_centerlines_cropped.geoms
                    if (l.bounds[3]-l.bounds[1]) > y_length_threshold] # _, miny, _, maxy = cline.bounds; the tolerance is in pixels
            Centerlines_clean_out = shapely.geometry.MultiLineString(Centerlines_clean)
        elif isinstance(Multi_centerlines_cropped, LineString):
            Centerlines_clean_out = Multi_centerlines_cropped.simplify(tolerance=simplification_tolerance, preserve_topology=False)
        else:
            logger.warning(f"Unexpected geometry after clipping: {Multi_centerlines_cropped.geom_type}")
            return



    logger.info("find_centerlines FINISH")
    return Centerlines_clean_out

#######################################################################
# Turn contours into lines and find nearest points between them for measure
#######################################################################
# Return paired point coordinates
## helper to find slope of each ring
"""    
elif intersection.geom_type == 'LineString':

    x, y = intersection.coords.xy
    x_dif = abs(x[-1] - x[0])

    if x_dif < frame_width * .20:  # This should be adjusted now it should skip this frame if a line is less then 20% of the frame width
        # print(i, 'th is too short')
        continue
    else:
        # print(i, "th frame is simple")
        slope, _, _, _, _ = scipy.stats.linregress(x, y)
        # logger.info("slope:{}".format(slope))
        if slope > 0 and slope < 2:
            PlusMinus = 1
        elif slope < 0 and slope > -2:
            PlusMinus = 0
        else:
            PlusMinus = []
        PlusMinus_index.append([PlusMinus, cut_point])
"""

def _find_ring_slopes(Multi_centerlines, imgheight, imgwidth):
    PlusMinus_index = []
    frame_width = imgheight * .5 #.75
    sliding = frame_width * .5  # How much is the frame sliding in every loop
    # print('frame_width', frame_width)
    number_of_segments = int(imgwidth / sliding)
    logger.debug("number_of_segments: %s", number_of_segments)
    # Slide by frames along the Multicenterline and evaluate the slope
    for i in range(number_of_segments):
        # print('loop_number', i)
        # get the frame
        frame_poly = shapely.geometry.box(i * sliding, 0, (i * sliding) + frame_width, imgheight)
        cut_point = i * sliding + (frame_width * .5)  # Better to get cutting point here and use instead of frame number
        logger.debug("frame_poly.exterior.coords.xy: %s", frame_poly.exterior.coords.xy)

        # get lines inside of the frame
        intersection = Multi_centerlines.intersection(frame_poly)
        logger.debug("intersection type prior: %s", intersection.geom_type)
        if intersection.is_empty or isinstance(intersection, Point):  # prevents crushing if segment is empty
            logger.info("Intersection is empty or contains only one point")
            continue

        else:
            if isinstance(intersection, LineString):
                logger.debug("Converting LineString to MultiLinestring")
                intersection = shapely.geometry.MultiLineString([intersection])

            logger.debug("intersection type after: %s", intersection.geom_type)
            slopes = []
            for line in intersection.geoms:
                #x, y = line.coords.xy
                minx, _, maxx, _ = line.bounds
                x_dif = maxx - minx
                # print('loop number and xy coords:',i, l, x, y)
                if x_dif < frame_width * .20:  # This can be adjusted now it should skip this frame if line is less than 20% of the frame width
                    # print(i, 'th is too short')
                    continue
                else:
                    # print(i, "th frame is complex")
                    #slope, _, _, _, _ = scipy.stats.linregress(x, y)
                    #print("slope", slope)
                    # new slope
                    coords = line.coords

                    x0, y0 = coords[0]
                    x1, y1 = coords[-1]

                    dx = x1 - x0
                    if abs(dx) < 1e-6:
                        logger.debug("dx: %s", dx)
                        continue

                    slope = (y1 - y0) / dx
                    #print("slope2", slope2)

                slopes.append(slope)

            if not slopes:
                continue

            mean_slopes = np.mean(slopes)

            if mean_slopes > 0 and mean_slopes < 2:
                PlusMinus = 1
            elif mean_slopes < 0 and mean_slopes > -2:
                PlusMinus = 0
            else:
                PlusMinus = []
            PlusMinus_index.append([PlusMinus, cut_point])

    return PlusMinus_index

def _find_cutting_point(PlusMinus_index, imgheight):
    test_seq1, test_seq2 = [0, 0, 1, 1], [1, 1, 0, 0]

    cutting_point = None
    PlusMinus = [x for x, _ in PlusMinus_index]
    for i in range(len(PlusMinus_index)):
        pm_seq = PlusMinus[i:i + len(test_seq1)]
        if pm_seq not in (test_seq1, test_seq2):
            continue
        if cutting_point is not None:
            log_and_print("Several cutting points identified, needs to be investigated!", logger, "warning")
            #print('Several cutting points identified, needs to be investigated!')
            #logger.warning('Several cutting points identified, needs to be investigated!')
            break
        cutting_point = PlusMinus_index[i + 1][1] + ((PlusMinus_index[i + 2][1] - PlusMinus_index[i + 1][1]) / 2)
        # if cutting_point is immediately at the beginning of the sample ignore it
        if cutting_point < imgheight * 2:  # if cutting point is within 2*image height it will be ignored
            logger.debug("cutting_point is at the beginning of the image and will be ignored")
            cutting_point = None

    return cutting_point

def _measure_distances(Multi_centerlines):
    # Reorder centerlines by x_middle
    """
    x_mins, x_maxs = [geom.bounds[0] for geom in Multi_centerlines.geoms], [geom.bounds[2] for geom in
                                                                             Multi_centerlines.geoms]

    x_middle = np.array(x_mins) + (np.array(x_maxs) - np.array(x_mins)) / 2
    """
    x_middle = [(geom.bounds[0] + geom.bounds[2]) / 2 for geom in Multi_centerlines.geoms]

    #contourszip = zip(x_middle, Multi_centerlines.geoms)

    centerlines = [x for _, x in sorted(zip(x_middle, Multi_centerlines.geoms), key=itemgetter(0))]
    Multi_centerlines = shapely.geometry.MultiLineString(centerlines)
    # print('ordered centerlines2:', Multi_centerlines2.geom_type)
    measure_points = tuple(nearest_points(Multi_centerlines.geoms[i], Multi_centerlines.geoms[i + 1]) for
                            i in range(len(Multi_centerlines.geoms) - 1))

    return Multi_centerlines, measure_points

def _cut_sections_and_measure(Multi_centerlines, cutting_point, imgheight, before_cutting_point=True):
    # Output is the _nearest_distances_output: Multi_centerlines, measure_points

    if before_cutting_point:
        cut_frame = shapely.geometry.box(0, 0, cutting_point, imgheight)
    else:
        cut_frame = shapely.geometry.box(cutting_point, 0, imgwidth, imgheight)

    geom = Multi_centerlines.intersection(cut_frame)

    if isinstance(geom, LineString):
        return None

    elif isinstance(geom, MultiLineString):
        return _measure_distances(geom)

    elif isinstance(geom, GeometryCollection):

        lines = []
        for g in geom.geoms:
            if isinstance(g, LineString):
                lines.append(g)
            elif isinstance(g, MultiLineString):
                lines.extend(g.geoms)

        if lines and len(lines) > 1:
            return _measure_distances(shapely.geometry.MultiLineString(lines))

    return None

def measure_contours(Multi_centerlines, image):
    logger.info("measure_contours START")
    imgheight, imgwidth = image.shape[:2]
    logger.info(f"Image has height {imgheight} and width {imgwidth}")
    logger.debug("%s ring boundaries were detected", len(Multi_centerlines.geoms))

    # Split samples that are crossing center into two then turn the second part around
    # Find the point where the sample is crossing a pith by the change in a slope of the lines in PlusMinus_index
    PlusMinus_index = _find_ring_slopes(Multi_centerlines, imgheight, imgwidth)
    cutting_point = _find_cutting_point(PlusMinus_index, imgheight)

    # Split sequence where it is crossing the middle
    if cutting_point:
        logger.info(f'Core sample crosses the center and is cut at: {cutting_point}')
        """
        cut_frame1_poly = shapely.geometry.box(0, 0, cutting_point, imgheight)
        Multi_centerlines1 = Multi_centerlines.intersection(cut_frame1_poly)
        cut_frame2_poly = shapely.geometry.box(cutting_point, 0, imgwidth, imgheight)
        Multi_centerlines2 = Multi_centerlines.intersection(cut_frame2_poly)
        """
        # Part before pith
        Multi_centerlines1, measure_points1 = _cut_sections_and_measure(Multi_centerlines, cutting_point, imgheight, True)
        # Part after pith
        Multi_centerlines2, measure_points2 = _cut_sections_and_measure(Multi_centerlines, cutting_point, imgheight, False)

        if Multi_centerlines1 is None:
            log_and_print("Multi_centerlines1 is empty", "warning")
            return None
        elif Multi_centerlines2 is None:
            logger.info("Multi_centerlines2, the part after cutting point, is only one line")
            measure_points = (measure_points1,)
            Multi_centerlines = (Multi_centerlines1,)
        else:
            return (Multi_centerlines1, Multi_centerlines2), (measure_points1, measure_points2), cutting_point

        """
        if Multi_centerlines2.geom_type=='LineString':
            logger.info("Multi_centerlines2, the part after cutting point, is only one line")
            measure_points = (measure_points1,)
            Multi_centerlines = (Multi_centerlines1,)

        else:
            Multi_centerlines2, measure_points2 = _measure_distances(Multi_centerlines2)

            measure_points = (measure_points1, measure_points2)
            Multi_centerlines = (Multi_centerlines1, Multi_centerlines2)
        """

        return Multi_centerlines, measure_points, cutting_point

    else:
        # Loop through them to measure pairwise distances between nearest points
        logger.info('Middle point was not detected')
        cutting_point = {}

        Multi_centerlines, measure_points = _measure_distances(Multi_centerlines)

        logger.info("measure_contours FINISH")
        return (Multi_centerlines,), (measure_points,), cutting_point

#######################################################################
# Plot predicted lines and points of measurements to visually assess
#######################################################################
def plot_lines(image, centerlines, measure_points, file_name, path_out, plot_dpi=100, line_width=2):
    # line_width bigger means thicker line
    # Create pngs folder in output path
    logger.info("plot_lines START")
    MAX_WIDTH = 30000
    export_path = os.path.join(path_out, 'pngs')

    os.makedirs(export_path, exist_ok=True)

    f = file_name + '.png'
    # Save images at original size unles they are bigger in px than length 30000. Should improve diagnostics on the images
    imgheight, imgwidth = image.shape[:2]
    # since I use cv2 to load image I need to convert it to RGB before plotting with matplotlib
    #print("image.dtype", image.dtype)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print('imgheight, imgwidth', imgheight, imgwidth)

    if imgwidth < MAX_WIDTH:
        plt.figure(figsize=(imgwidth/plot_dpi, 2*(imgheight/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
        linewidth = (imgheight/1000)*line_width   # looks very variable depending on the image resolution when set as a constant
    else:  # adjust image size if it`s exceeding 30000 pixels to 30000
        resized_height = imgheight*(MAX_WIDTH/imgwidth)
        plt.figure(figsize=(MAX_WIDTH/plot_dpi, 2*(resized_height/plot_dpi)), dpi=plot_dpi)
        #fig, (ax1, ax2) = plt.subplots(2)
        plt.imshow(image)
        linewidth = (resized_height/1000)*line_width  # looks very variable depending on the image resolution when set as a constant

    if centerlines:
        # Plot the lines to the image
        color = ['g', 'b']
        for l in range(len(centerlines)):
            # define centerlines1 as a linestring in both cases if centerlines is Linestring or multilinestring
            geom = centerlines[l]
            logger.debug("centerlines[l].geom_type: %s", geom.geom_type)
            if isinstance(geom, MultiLineString):
                centerlines1 = geom.geoms
            else:
                centerlines1 = centerlines
            logger.debug("centerlines1: %s", centerlines1)

            for i, centerline in enumerate(centerlines1):
                logger.debug("centerline: %s", centerline)

                xc, yc = centerline.coords.xy
                plt.plot(xc, yc, color[l], linewidth=linewidth)

                if measure_points:
                    measure_points1 = measure_points[l]
                    logger.debug("measure_points1: %s", measure_points1)
                    if len(measure_points1) == 0:  # Precaution in case the first part of measure points is empty
                        continue

                    if i < len(measure_points1):  # there is one less measure points than lines
                        points = measure_points1[i]
                        xp, yp = points[0].coords.xy
                        xp1, yp1 = points[1].coords.xy
                        plt.plot([xp, xp1], [yp, yp1], 'r', linewidth=linewidth)

    plt.savefig(os.path.join(export_path, f), bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info("plot_lines FINISH")
#######################################################################
# Create a JSON file for shiny app
#######################################################################
def write_to_json(image_name, cutting_point, run_ID, path_out, centerlines_rings,
                    clean_contours_rings, clean_contours_cracks=None):
    logger.info("write_to_json START")
    # Define the structure of json
    out_json = {image_name: {'run_ID':run_ID, 'predictions':{}, 'directionality': {},
                            'center': {}, 'est_rings_to_pith': {}, 'ring_widths': {}}}
    out_json[image_name]['predictions'] = {'ring_line': {}, 'ring_polygon': {},
                                            'crack_polygon': {}, 'resin_polygon': {},
                                            'pith_polygon': {}, 'version': '2.0.0'}
    out_json[image_name]['center'] = {'cutting_point': cutting_point, 'pith_present': {},
                                        'pith_inferred': {'coords': {'x': [], 'y': []}}}
    out_json[image_name]['ring_widths'] = {'directionality': {}, 'shortest_distance': {},
                                            'manual': {}}
    # Separate x and y coordinates for polygons and line
    if clean_contours_cracks is None or len(clean_contours_cracks) == 0:
        input_vars = (centerlines_rings, shapely.multipolygons(clean_contours_rings))
    else:
        logger.debug("clean_contours_rings length: %s", len(clean_contours_rings))
        logger.debug("clean_contours_cracks length: %s", len(clean_contours_cracks))
        input_vars = (centerlines_rings, shapely.multipolygons(clean_contours_rings), shapely.multipolygons(clean_contours_cracks))

    logger.debug("input_vars length: %s", len(input_vars))
    json_names = ('ring_line', 'ring_polygon', 'crack_polygon')
    predictions = out_json[image_name]['predictions'] # catch for faster assigning in the loop
    for json_name, input_var in zip(json_names, input_vars):
        logger.debug("input_var %s", input_var)
        logger.debug("json_name: %s", json_name)
        coords = {}

        for geom in input_var.geoms:
            logger.debug("geom %s", geom)
            logger.debug("geom type: %s", geom.geom_type)
            if isinstance(geom, Polygon):
                geom = geom.exterior
            x_list, y_list = geom.coords.xy
            #x_list = list(map(int, x_list))
            #y_list = list(map(int, y_list))
            x_list = np.asarray(x_list, dtype=np.int32).tolist()
            y_list = np.asarray(y_list, dtype=np.int32).tolist()

            #print('x_list', x_list)
            # now add everything in the json
            x_min = min(x_list)
            the_coord = str(x_min) + '_' + 'coords'
            logger.debug("the_coord: %s", the_coord)
            #coords[the_coord] = {}
            #coords[the_coord]['x'] = x_list
            #coords[the_coord]['y'] = y_list
            coords[the_coord] = {'x': x_list, 'y': y_list}
            # print("coords",type(coords))

        #print("coords",type(coords))
        predictions[json_name] = coords

    output = os.path.join(path_out, os.path.splitext(image_name)[0] + '.json')
    with open(output, 'w') as outfile:
        ujson.dump(out_json, outfile)
    logger.info("write_to_json FINISH")
#######################################################################
# Create a POS file with measure points
#######################################################################
def _point_to_string(point, mm_per_pixel):
    x, y = point.coords[0]
    return f"{x*mm_per_pixel:.3f},{y*mm_per_pixel:.3f}"

def write_to_pos(measure_points, file_name, image_name, DPI, path_out):
    logger.info("write_to_pos START")
    # If two adjust naming. Nothing for the normal one and add "x" at the end for the second part
    # Prepare date, time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    # Prepare unit conversion
    mm_per_pixel = 25.4 / DPI
    # Create paths for output files
    out_file_path, out_fileX_path = os.path.join(path_out, file_name+".pos"), os.path.join(path_out, file_name+"X"+".pos")
    out_file_paths = (out_file_path, out_fileX_path)

    logger.debug("measure_points %s", measure_points)
    for l, measure_points1 in enumerate(measure_points):
        logger.debug("measure_points1 %s", measure_points1)
        logger.debug("len of measure_points1: %s", len(measure_points1))
        if len(measure_points1) == 0:  # Precaution in case the first part of measure points is empty
            logger.warning("Middle of the core identified on the first ring!!!Only X .pos file will be created!!!")
            continue
        str_measure_points1 = []
        logger.debug("measure_points1[0][0]: %s", measure_points1[0][0])
        # The first point
        str_measure_points1.append(_point_to_string(measure_points1[0][0], mm_per_pixel) + "\n")
        # The middle points
        for i in range(len(measure_points1)-1):
            # This gets second point of a current tuple and the first of the next tuple
            str_measure_points1.append(_point_to_string(measure_points1[i][1], mm_per_pixel) + "  "
                                       + _point_to_string(measure_points1[i+1][0], mm_per_pixel) + "\n")
        # The last point
        logger.debug("should be last measure point %s", len(measure_points1))
        str_measure_points1.append(_point_to_string(measure_points1[len(measure_points1)-1][1], mm_per_pixel) + "\n")

        # Write in the file

        with open(out_file_paths[l], 'w') as f:
            f.write(f'#DENDRO (Cybis Dendro program compatible format) Coordinate file written as \n'
                    f'#Imagefile {image_name} \n'
                    f'#DPI {DPI} \n'
                    f'#All coordinates in millimeters (mm) \n'
                    f'SCALE 1 \n'
                    f'#C DATED \n'
                    f'#C Written={dt_string} \n'
                    f'#C CooRecorder= \n'
                    f'#C licensedTo=; \n')

            f.write("".join(str_measure_points1))

    logger.info("write_to_pos FINISH")
