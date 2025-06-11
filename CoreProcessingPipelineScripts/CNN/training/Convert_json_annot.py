import os
import json

DATASET_PATH = '/Users/miroslav.polacek/Github/TRG_YOLOv8_try/training/sample_dataset'

def convert_annot(annotations_json):


######## DEVELOPMENT

subset = 'val'

FOLDER_PATH = os.path.join(DATASET_PATH, subset)
JSON_PATH = os.path.join(FOLDER_PATH, 'via_region_data_transformed.json')

with open(JSON_PATH, 'r') as file:
    json_dict = json.load(file)

for _ , v in json_dict.items():
    # level of individual image save txt at this level after I have all values
    filename = v['filename']
    # create a text file named filename.txt
    annot_txt_file = os.path.join(FOLDER_PATH, os.path.splitext(filename)[0] + ".txt")
    f = open(annot_txt_file, "a+")

    im_width, im_height = v['size'].split('x')
    for region in v['regions']:
        class_name = region['region_attributes']['type']
        all_points_x = region['shape_attributes']['all_points_x']
        all_points_y = region['shape_attributes']['all_points_y']
        # normalize the coordinates by image size
        all_points_x_norm = [x/int(im_width) for x in all_points_x]
        all_points_y_norm = [y/int(im_height) for y in all_points_y]
        # make a row such as: class_code x1 y1 x2 y2
        if class_name == "RingBndy":
            line = "\n0 "
        elif class_name == "Crack":
            line = "\n1 "
        else:
            print(f"Class name neither RingBndy nor Crack in image {filename}, it is {class_name}")
            continue

        for i in range(len(all_points_x_norm)):
            line += str(all_points_x_norm[i]) + " "
            line += str(all_points_y_norm[i]) + " "
        f.write(line)
    f.close()


