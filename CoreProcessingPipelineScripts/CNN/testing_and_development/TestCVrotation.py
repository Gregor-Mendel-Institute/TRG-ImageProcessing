import os
import cv2
import skimage
import numpy as np
import time

image_path = "/Users/miroslav/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/train/14610_00014006b_0_pSX1.9653764466903185_pSY1.9665786978105748.tif"
cropped_part = cv2.imread(image_path)

############# Functions #############
#CV2
start = time.perf_counter()
rot45 = _rotate_image(cropped_part, 45)
end = time.perf_counter()
print("run_time", end - start) # 0.05526287299289834, 0.022680124995531514, 0.016969449003227055
mean_CV = np.mean([0.05526287299289834, 0.022680124995531514, 0.016969449003227055])

rot_back = _rotate_image(rot45, -45)
#skimage


skim_start = time.perf_counter()
skim45 = skimage.transform.rotate(cropped_part, angle=45,
                                               preserve_range=True, resize=True).astype(np.uint8)
skim_end = time.perf_counter()
print("skim", skim_end - skim_start) # 1.0920688780024648, 0.6973726900032489, 0.8532868290058104
mean_SK = np.mean([1.0920688780024648, 0.6973726900032489, 0.8532868290058104])