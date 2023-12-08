import os
import sys
#import itertools
#import math
#import logging
#import json
#import re
#import random
#from collections import OrderedDict
import numpy as np
import matplotlib
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#import matplotlib.lines as lines
#from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN/")
# ROOT_DIR = os.path.abspath("./CoreProcessingPipelineScripts/CNN/Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize_print
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from training.retraining_container import TreeringConfig, TreeringDataset

%matplotlib inline

# Load configurations
config = TreeringConfig()
DATASET_PATH = os.path.join(ROOT_DIR, "training/retrain_testing_dataset")

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = TreeringDataset()
dataset.load_treering(DATASET_PATH, "train")

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize_print.display_top_masks(image, mask, class_ids, dataset.class_names)