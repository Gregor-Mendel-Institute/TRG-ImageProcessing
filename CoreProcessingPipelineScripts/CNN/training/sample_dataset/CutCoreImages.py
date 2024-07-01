# Chop core images into squares to prepare them for annotation
import cv2
import os

# path to folder with cores to be chopped
core_folder_path = '/Volumes/T7 Shield/CNN_yolov8_retraining_data/OurImages_raw/'
# search location and make a list
im_names_list = [f for f in os.listdir(core_folder_path) if not f.startswith('.') and f.endswith('.tif')]
# Append to make image_path

#image_path = '/Volumes/T7 Shield/CNN_yolov8_retraining_data/OurImages_raw/00019005b_mo4577_pS1.96536799834280303030.tif'
out_path = '/Volumes/T7 Shield/CNN_yolov8_retraining_data/OurImages_raw/chopped'
# start function here
for im_name in im_names_list:
    image_path = os.path.join(core_folder_path, im_name)
    im = cv2.imread(image_path)
    imheight, imwidth = im.shape[:2]
    for i in range(0, imwidth, imheight):
        print(i)
        im_crop = im[:, i:i+imheight, :]
        # get file name
        #file_name = os.path.basename(image_path)
        file_name_no_ext = os.path.splitext(im_name)[0]
        im_out_path = os.path.join(out_path, file_name_no_ext + '_' + str(i) + '.tif')
        cv2.imwrite(im_out_path, im_crop)
