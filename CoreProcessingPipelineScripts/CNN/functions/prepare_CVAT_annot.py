"""
Prepares annotations from .xml files from CVAT_annotations folder and distributes them based on the image names in
train and val folders.
In the CVAT annotation file the ring boundary has to be a polyline RingBndy and crack as polygone labeled CrackPoly.
"""

############# IMPORTS #################
import os
from lxml import etree
from shapely.geometry import LineString, Polygon, box

#########################################################
# FUNCTIONS
#########################################################
def collect_annotations(CVAT_folder):
    # extracts individual image annotations dictionaries from all xml files and put them in one list
    all_image_xml_list = []
    CVAT_annot_file_list = [f for f in os.listdir(CVAT_folder) if not f.startswith('.')]  # ignore hidden files
    # make sure they are xml files
    for xml_name in CVAT_annot_file_list:
        if xml_name.endswith('.xml'):
            # load xml file
            print("Loading annotation file", xml_name)
            with open(os.path.join(CVAT_folder, xml_name)) as file:
                root = etree.parse(file).getroot()

            for i in root.iter('image'):
                all_image_xml_list.append(i)
        else:
            print(f"{xml_name} not proccesed because it`s not valid annotation file")

    return all_image_xml_list

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

def preparepolygon(polygon_str):
    # Output x and y coords of polygone.
    points = polygon_str.split(";")
    xy_coords = list(tuple(map(float, (point.split(",")))) for point in points)
    polygon = Polygon(xy_coords)
    x, y = polygon.exterior.coords.xy

    return list(x), list(y)

def prepare_annotations(dataset_path, annot_list, buffer=10, overwrite_existing=False):
    dir_list = os.listdir(dataset_path)
    supported_extensions = ['.tif', '.tiff']
    im_list = [f for f in dir_list if os.path.splitext(f)[1] in supported_extensions]

    for i in annot_list:
        # get image name to control the loop
        [image_name] = [value for key, value in i.items() if key == "name"]
        annot_txt_file = os.path.join(dataset_path, os.path.splitext(image_name)[0] + ".txt")
        if image_name not in im_list:
            continue
        elif os.path.isfile(annot_txt_file) and overwrite_existing==False:
            continue
        else:
            print("Processing annotation file of image", image_name)
            # create a text file named filename.txt
            f = open(annot_txt_file, "w+")
            for key, value in i.items():
                if key == "width":
                    im_width = value
                if key == "height":
                    im_height = value

            # get polylines which are Ring annotations
            for poly_tag in i.iter('polyline'):
                print(poly_tag)
                for key, value in poly_tag.items():
                    # print(key)
                    # print(value)
                    if key == "label":
                        # print(value)
                        if value != "RingBndy":
                            print("Warning: Label is not RingBndy. Continue assuming all polylines are rings")
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
                print(poly_tag)
                for key, value in poly_tag.items():
                    # print(key)
                    # print(value)
                    if key == "label":
                        # print(value)
                        if value != "CrackPoly":
                            print("Warning: Label is not CrackPoly. Continue assuming all polygon are crack")
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

######## MAIN ############################
def prepare_all_annotations(dataset_path, buffer=10, overwrite_existing=True):

    CVAT_ANNOT_PATH = os.path.join(dataset_path, 'CVAT_annotations')
    all_annot_list = collect_annotations(CVAT_folder=CVAT_ANNOT_PATH)

    subset_list = ['train', 'val']
    for subset in subset_list:
        FOLDER_PATH = os.path.join(dataset_path, subset)
        prepare_annotations(dataset_path=FOLDER_PATH, annot_list=all_annot_list, buffer=buffer, overwrite_existing=overwrite_existing)

###### TESTING ##########
# DATASET_PATH = '/Users/miroslav.polacek/Github/TRG_YOLOv8_try/training/sample_dataset' # for development
# DATASET_PATH = '/Volumes/swarts/user/miroslav.polacek/FullSpruceDatasetWithCVAT5pxBuffer'
# prepare_all_annotations(dataset_path=DATASET_PATH, buffer=5, overwrite_existing=True)

