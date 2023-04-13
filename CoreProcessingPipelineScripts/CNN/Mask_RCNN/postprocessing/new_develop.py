import pandas as pd
import skimage.io
from skimage import exposure, img_as_ubyte

# import numpy as np

image_path = "../TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/training/sample_dataset/train/1350_00041008a_0_pSX1.965402638101432_pSY1.9654116736824034.tif"
image_path = "/Users/miroslav.polacek/Downloads/C67A_short.tif"  # RGBA 8bit
image_path = "/Users/miroslav.polacek/Downloads/C67A.tif"  # RGB 8bit
image_path = "/Users/miroslav.polacek/Downloads/Q3L51A_short_8bit.tif"  # RGB 8bit
image_path = "/Users/miroslav.polacek/Downloads/Q3L51A_skippy_original.tif"  # seems RGBA 8bit

im_origin = skimage.io.imread(image_path)
im_8bit = img_as_ubyte(exposure.rescale_intensity(im_origin))

# %%
# check number of channels and if 4 assume rgba and convert to rgb
# conversion if image is not 8bit convert to 8 bit
if im_origin.dtype == 'uint8' and im_origin.shape[2] == 3:
    print("Image was 8bit and RGB")
elif im_origin.shape[2] == 4:
    print("Image has 4 channels, assuming RGBA, trying to convert")
    im_origin = img_as_ubyte(skimage.color.rgba2rgb(im_origin))
elif im_origin.dtype != 'uint8':
    print("Image converted to 8bit")
    im_origin = img_as_ubyte(exposure.rescale_intensity(im_origin))  # with rescaling should be better
# %%

### SOLVE THE ISSIU WITH SORT
import pickle
import numpy as np
import random
contours_filtered_notOK_path = "CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/DebugContoursFiltered_Natalia_NotOk.pkl"
contours_filtered_OK_path = "CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/DebugContoursFiltered.pkl"


with open(contours_filtered_notOK_path, 'rb') as f:
    [contours_filtered_notOK, x_mins] = pickle.load(f)

contourszip = zip(x_mins, contours_filtered_notOK)
contours_out = [x for _, x in sorted(contourszip, reverse=False)]

####### EXPERIMENT
random.shuffle(x_mins)
x_mins = list(x_mins[i] for i in [1, 106])
contours_filtered_notOK = list(contours_filtered_notOK[i] for i in [1, 106])
###########

# Explore the data
exp_data_pd = pd.DataFrame(columns=["x_mins", "contour_0"])
for i in range(len(x_mins)):
    x_min = x_mins[i]
    contour_0 = contours_filtered_notOK[i].shape[0]
    print(x_min, contour_0)
    #exp_data_pd = pd.DataFrame(np.array([[x_min], [contour_0]]).T, columns=["x_mins", "contour_0"])
    to_concat = pd.DataFrame(np.array([[x_min], [contour_0]]).T, columns=["x_mins", "contour_0"])
    exp_data_pd = pd.concat([exp_data_pd, to_concat], axis=0, ignore_index=True)

####### TEST SHAPELY #######
import pickle
import shapely

Multi_centerlines_path = "/Users/miroslav.polacek/Github/TRG_Tf2/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/Muliti_centerlines.pkl"
with open(Multi_centerlines_path, 'rb') as f:
    Multi_centerlines = pickle.load(f)