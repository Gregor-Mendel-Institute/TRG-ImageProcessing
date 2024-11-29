"""
The function should take all the images in given folder, print pngs with annotations in "annot_check" subdirectory, on
the way write in a log summary of annotations: how many images, how many rings and how many cracks, list images that
do not have cracks or rings and images that do not have anny annotations save a text file with all of this in
the subdirectory as well.
"""
import os
import numpy as np
import time
# Import Mask RCNN
ROOT_DIR = os.path.abspath('./CoreProcessingPipelineScripts/CNN/') # to run in Pycharm
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library
from functions.processing_functions import check_annot_dataset

# testing on sample dataset
start_time = time.perf_counter()
dataset = "/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/"
check_annot_dataset(dataset)
finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}")


# run on my dataset
dataset = "/Volumes/T7 Shield/TRG_RingCrack_3rd_trainingdataset"
check_annot_dataset(dataset)



### testing in converting all lists in tuples in a finction would make difference but it seems as soon as I concatanate tuples its taking longer
im_size = (1200, 1300)
annot_path = '/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/train/14610_00014006b_0_pSX1.9653764466903185_pSY1.9665786978105748.txt'
def load_annot_l(annot_path, im_size):
    # annot_path is path to annotation txt file of yolov8 format
    # im_size is output of .shape method
    # output contours are in shape acceptable for cv2 contours
    # load annotations
    labels, contours = [], []
    with open(annot_path, "r") as f:
        for line in f:
            if len(line) < 3:  # it has to be much more to be valid points but in my case its sometimes a space in a row
                continue
            label, annot_list = line.split()[0], line.split()[1:]
            labels.append(label)
            annot_list = list(map(float, annot_list))  # convert x,y from string to a value
            #annot_list = [int(i*im_size) for i in annot_list]
            annot_list_xy = []
            i = 0
            while (i < len(annot_list)):
                # append in the coordinates as a [x,y] plus convert them to pixels
                annot_list_xy.append([annot_list[i]*im_size[1], annot_list[i + 1]*im_size[0]])
                i += 2
            contours.append(np.array(annot_list_xy, dtype=np.int32))
    return contours, labels


def load_annot_t(annot_path, im_size):
    # annot_path is path to annotation txt file of yolov8 format
    # im_size is output of .shape method
    # output contours are in shape acceptable for cv2 contours
    # load annotations
    labels, contours = (), ()
    with open(annot_path, "r") as f:
        for line in f:
            if len(line) < 3:  # it has to be much more to be valid points but in my case its sometimes a space in a row
                continue
            label, annot_list = line.split()[0], line.split()[1:]
            labels += (label,)
            annot_list = list(map(float, annot_list))  # convert x,y from string to a value
            #annot_list = [int(i*im_size) for i in annot_list]
            annot_list_xy = []
            i = 0
            while (i < len(annot_list)):
                # append in the coordinates as a [x,y] plus convert them to pixels
                annot_list_xy.append([annot_list[i]*im_size[1], annot_list[i + 1]*im_size[0]])
                i += 2
            contours +=tuple(np.array(annot_list_xy, dtype=np.int32))
    return contours, labels

start_time = time.perf_counter()
c, l = load_annot_l(annot_path, im_size)
finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}")

start_time = time.perf_counter()
c, l = load_annot_t(annot_path, im_size)
finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}")

start_time = time.perf_counter()
i = 0
data = []
for i in range(1000000):
    print(i)
    data.append('relatively long string')
    i += 1
print(data)
finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}")

start_time = time.perf_counter()
#data = ('relatively long string', 'relatively long string', 'relatively long string', 'relatively long string', 'relatively long string')
#data = tuple(data)
new_string = [i for i in data]
"""
for i in data:
    new_string.append(i)
"""
print(new_string)

finished_time = time.perf_counter()
print(f"Total time: {str(finished_time - start_time)}")