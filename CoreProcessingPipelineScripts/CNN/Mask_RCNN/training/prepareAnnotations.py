#!/usr/bin/env python
"""
Prepare CVAT annotations from .xml files. Polylines are transformed into polygons.
The annotations are organised based on images in train and val folders and saved as a .json file.
"""

import os
import json
from lxml import etree
from shapely.geometry import LineString, box
#from shapely import LineString # this will work in newer versions of shapely
import numpy as np
import copy

#################################################################################
# FUNCTIONS
####################################
def polylinetopolygon(polyline_str, width, height, buffer=30):
    # takes polyline string form CVAT xml.
    # Using shapely package it transforms line into polygon
    # Output x and y coords of polygone.
    # width and height of the image ensure that polygone is not exceeding image
    points = polyline_str.split(";")
    #coords = list((x, y) for point.split(",") in points)
    xy_coords = list(tuple(map(float, (point.split(",")))) for point in points)
    polyline = LineString(xy_coords)
    im_box = box(1, 1, int(width)-1, int(height)-1) # to be sure i crop it one pixel inside the image
    polygon_ring = polyline.buffer(buffer)
    polygon_clean = polygon_ring.intersection(im_box)
    x, y = polygon_clean.exterior.coords.xy

    return list(x), list(y)

def pocessCVATxml(dataset):
    buffer = 30
    newJ = {}
    annotated_images = []
    shpAttrBase = {'name': 'polygon'}
    CVAT_ANNOTPATH = os.path.join(dataset, "CVAT_annotations")
    CVAT_annot_file_list = os.listdir(CVAT_ANNOTPATH)
    for xml_name in CVAT_annot_file_list:
        # open load xml file
        with open(os.path.join(CVAT_ANNOTPATH, xml_name)) as file:
            root = etree.parse(file).getroot()
        # Initiate new dictionary to add images and annotations to it
        for i in root.iter('image'):
            # get image name
            for key, value in i.items():
                #print(key, value)
                if key == "name":
                    image_name = value
                    print("Processing annotation file of image", image_name)
                    annotated_images.append(image_name)
                if key == "width":
                    width = value
                if key == "height":
                    height = value
            image_size = str(width) + "x" + str(height)
            newJ[image_name] = {"file_attributes": {},
                                "filename": image_name,
                                "regions": [],
                                "size": image_size}
            # get polylines
            shpAtt = shpAttrBase.copy()
            for poly_tag in i.iter('polyline'):
                for key, value in poly_tag.items():
                    #print(key)
                    #print(value)
                    if key == "label":
                        #print(value)
                        if value != "RingBndy":
                            print("Warning: Label is not RingBndy. Continue assuming all polylines are rings ")
                    if key == "points":
                        all_points_x, all_points_y = polylinetopolygon(polyline_str=value, width=width, height=height, buffer=buffer)
                        shpAtt['all_points_x'] = all_points_x
                        shpAtt['all_points_y'] = all_points_y
                        # print('newAtt: ',shpAtt)
                        regions = {'shape_attributes': shpAtt, 'region_attributes': {'type': 'RingBndy'}}
                        newJ[image_name]['regions'].append(copy.deepcopy(regions))
    return newJ, annotated_images

def annottoimages(image_folder, annotation_file):
    # subset annotations for images in the folder
    # report missmatches
    # save the annotation json
    dict_out = {}
    dir_list = os.listdir(image_folder)
    supported_extensions = ['.tif', '.tiff']
    im_list = [f for f in dir_list if os.path.splitext(f)[1] in supported_extensions]

    for image_name in im_list:
        try:
            annotations = annotation_file[image_name]
            dict_out[image_name] = annotations
        except:
            print("There are no annotations for image", image_name)

    # Save the correct subset of annotations as a json
    JSON_OUT_PATH = os.path.join(image_folder, "via_region_data_transformed.json")
    with open(JSON_OUT_PATH, 'w') as outfile:
        json.dump(dict_out, outfile, indent=4)

    return dict_out

#################################################################################
# PROCESS
####################################

#DATASETPATH = "/Users/miroslav.polacek/Github/TRG_retraining/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/training/retrain_testing_dataset"
def prepareAnnotations(dataset, overwrite_existing=False):

    TRAIN_PATH = os.path.join(dataset, "train")
    VAL_PATH = os.path.join(dataset, "val")

    folder_paths =[TRAIN_PATH, VAL_PATH]
    # process and unite all xml files and prepare json with all annotations
    all_annotations, annotated_images = pocessCVATxml(dataset=dataset)

    # prepare annotations
    for folder in folder_paths:
        # Check if json already exists
        annot_json = os.path.join(folder, "via_region_data_transformed.json")
        if os.path.isfile(annot_json) and overwrite_existing==False:
            print(f'Annotation exists in {os.path.basename(folder)} folder.')
            continue
        else:
            _ = annottoimages(image_folder=folder, annotation_file=all_annotations)



