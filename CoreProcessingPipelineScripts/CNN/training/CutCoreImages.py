# Chop core images into squares to prepare them for annotation
import cv2
import os
import argparse

# path to folder with cores to be chopped
core_folder_path = '/Volumes/T7 Shield/CNN_yolov8_retraining_data/Plot23_badsamplequality/'
out_path = '/Volumes/T7 Shield/CNN_yolov8_retraining_data/Plot23_badsamplequality/chopped'
#######################################################################
# Arguments
#######################################################################
def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Segmentation of whole core')

    ## Compulsory arguments
    parser.add_argument('--im_to_crop', required=True,
                        help="Folder path with images that need top be cropped")

    ## Compulsory arguments
    parser.add_argument('--save_out_squares', required=True,
                        type=str,
                        help="Location to save image squares")
    args = parser.parse_args()
    return args

#######################################################################
# main
#######################################################################
def main():
    # get the arguments
    args = get_args()
    print(args)
    # search location and make a list
    supported_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    im_names_list = [f for f in os.listdir(args.im_to_crop) if not f.startswith('.') and f.endswith(supported_extensions)]
    # prepare output dir
    if not os.path.isdir(args.save_out_squares):
        os.mkdir(args.save_out_squares)
    # start function here
    for im_name in im_names_list:
        image_path = os.path.join(args.im_to_crop, im_name)
        im = cv2.imread(image_path)
        imheight, imwidth = im.shape[:2]
        for i in range(0, imwidth, imheight):
            print(i)
            im_crop = im[:, i:i+imheight, :]

            # get file name
            #file_name = os.path.basename(image_path)
            file_name_no_ext = os.path.splitext(im_name)[0]
            im_out_path = os.path.join(args.save_out_squares, file_name_no_ext + '_' + str(i) + '.tif')
            cv2.imwrite(im_out_path, im_crop)

if __name__ == '__main__':
    main()