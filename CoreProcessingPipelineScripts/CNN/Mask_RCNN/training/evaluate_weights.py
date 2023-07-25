"""
Complex evaluation of weights.

# Test on laptop
cd ~/Github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing &&
conda activate TreeRingCNN &&
bash eval_weight_realPost_laptopDebug.sh
set dataset path for debug
dataset

"""

#######################################################################
# Prepare packages, models and images
#######################################################################
import os
import sys
import skimage
import cv2
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Loading a preparing the model should be identical with the postprocessing file and does not need to be here
# Root directory of the project
# Import Mask RCNN
ROOT_DIR = os.path.abspath("../")
# ROOT_DIR = os.path.abspath("CoreProcessingPipelineScripts/CNN/Mask_RCNN/")
print('ROOT_DIR', ROOT_DIR)
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib
from mrcnn import utils
from DetectionConfig import TreeRing_onlyRing

from postprocessing.postprocessingCracksRings import sliding_window_detection_multirow, clean_up_mask, apply_mask, plot_lines

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Prepare ring model
config = TreeRing_onlyRing.TreeRingConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

#################################################################################
# Precision and recall for mask, first value of TP...should be for score of 0.5
#################################################################################
def TP_FP_FN_per_score_mask(gt_mask, pred_mask, scores, IoU_treshold):

    #loop scores
    score_range = np.arange(0.5, 1.0, 0.05)

        #print(gt_r)
        #print(pred_r)
    gt_rings = []
    pred_rings = []
    TPs = []
    FPs = []
    FNs = []
    #print("GT_MASK_SHAPE", gt_mask.shape)
    for SR in score_range:
        #print(SR)
        score_ids = np.where(scores > SR)[0] #Ids for predictions above certain score threshold
        #print(score_ids)
        mask_SR = np.take(pred_mask, score_ids, axis=2)
        #print('mask_SR.shape:', mask_SR.shape)

        mask_matrix = utils.compute_overlaps_masks(gt_mask, mask_SR)
        #print("mask_matrix", mask_matrix)
        #for every score range callculate TP, ...append by the socre ranges
        # making binary numpy array with IoU treshold
        mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
        #print (mask_matrix_binary)

        #GT rings and predicted rigs
        #print("MASK MATRIX SHAPE", mask_matrix.shape)

        if mask_matrix.shape[0]==0:
            TPs.append(0)
            FPs.append(mask_SR.shape[-1]) # All predicted are false in this case
            FNs.append(0)
        else:
            gt_r = len(mask_matrix)
            pred_r = len(mask_matrix[0])
            #TP
            sum_truth = np.sum(mask_matrix_binary, axis=1)
            sum_truth_binary = np.where(sum_truth > 0, 1, 0)
            TP = np.sum(sum_truth_binary)
            TPs.append(TP)
            #print('TPs:', TPs)
            #FP
            sum_pred = np.sum(mask_matrix_binary, axis=0)
            sum_pred_binary = np.where(sum_pred > 0, 1, 0)
            FP = pred_r - np.sum(sum_pred_binary)
            FPs.append(FP)
            #print('FP:', FP)
            #FN
            FN = gt_r - TP
            FNs.append(FN)

        #print('TPs:', TPs)
        #print('FPs:', FPs)
        #print('FNs:', FNs)
    #put together and sum up TP...per range

    return TPs, FPs, FNs, score_range

#######################################################################
# mAP graph
#######################################################################
def compute_ap_range_list(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    APlist = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            utils.compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask, iou_threshold=iou_threshold)
        APlist.append(ap)

        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))

    #print(APlist)
    return APlist
##########################################################################
# Turn flat combined mask into array with layer per every mask
##########################################################################
def modify_flat_mask(mask):
    #### identify polygons with opencv
    uint8binary = mask.astype(np.uint8).copy()

    # Older version of openCV has slightly different syntax i adjusted for it here
    if int(cv2.__version__.split(".")[0]) < 4:
        _, contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print('contour_shape:', len(contours))
    #print('contours.shape:', contours)
    #### in a loop through polygons turn every one into binary mask of propper dimension and append
    imgheight, imgwidth = mask.shape[:2]

    clean_contours = []
    for i in range(len(contours)):
        #print('contours[i]:',contours[i])
        #remove too small contours because they do not make sence
        rect = cv2.minAreaRect(contours[i])
        dim1, dim2 = rect[1]
        dim_max = max([dim1, dim2])
        if dim_max > imgheight/3:
            clean_contours.append(contours[i])

    #print('len_clean_contours', len(clean_contours))

    # create empty mask
    result_mask = np.zeros([imgheight, imgwidth, len(clean_contours)], dtype=np.uint8)

    for i in range(len(clean_contours)):
        #print('i', i)
        #print('len clean contours[i]', len(clean_contours[i]))
        x_points = []
        y_points = []
        for j in range(len(clean_contours[i])):
            #print('contij:', clean_contours[i][j])
            [[xc, yc]] = clean_contours[i][j]
            x_points.append(xc)
            y_points.append(yc)
        #print('len x', len(x_points))
        # Get indexes of pixels inside the polygon and set them to 1
        #print('x:', x_points)
        #print('y:', y_points)
        rr, cc = skimage.draw.polygon(y_points, x_points)
        #print('rr', rr)
        result_mask[rr, cc, i] = 1
        #print('res_mask', result_mask[:,:,0])


    return result_mask
##########################################################################
# Turn contours in binary mask
##########################################################################
def contours_to_binary(clean_contours,imheight,imwidth, debug=False):
    mask = np.zeros([imheight,imwidth, len(clean_contours)],
                    dtype=np.uint8)
    for i in range(len(clean_contours)):
        # separate x and y coords for contour
        x_list = []
        y_list = []
        for p in range(len(clean_contours[i])):
            [[x,y]] = clean_contours[i][p]
            x_list.append(x)
            y_list.append(y)

        r, c = skimage.draw.polygon(y_list, x_list)
        mask[r, c, i] = 1
        if debug==True:
            plt.imshow(mask[:,:,i])
            plt.show()
    if debug == True:
        flat_mask = mask.sum(axis=2)
        plt.imshow(flat_mask)
        plt.show()

    return mask

###########################################################################
# Calculate AP gorup of indexes. General and per class.
###########################################################################
def mAP_group(image, gt_class_id, gt_bbox, gt_mask, pred_bbox, pred_mask, pred_class_id, pred_scores):
    AP_general = []
    AP_names = ["mAP", "AP50", "APlist","mAP_ring", "AP50_ring", "APlist_ring","mAP_crack", "AP50_crack", "APlist_crack","mAP_resin", "AP50_resin", "APlist_resin","mAP_pith", "AP50_pith", "APlist_pith" ]
    # if no mask is detected
    #if pred_mask.shape[-1] == 0:
        #AP_general = [0,0,[0]*10]*5
        #print("mAP_group gave zeroes for this image")
    #else:

    # mAP, AP50 for all classes
    AP_list = compute_ap_range_list(gt_bbox, gt_class_id, gt_mask, pred_bbox, pred_class_id, pred_scores, pred_mask, verbose=0)
    mAP = np.array(AP_list).mean()
    #print("mAP for this image", mAP)
    AP50 = AP_list[0]
    AP_general = [mAP, AP50, AP_list]
    # for each class_id
    for i in range(1,5):
        #print("LOOP START", i)
        if gt_mask[:,:,gt_class_id==i].shape[-1] > 0:
            AP_list = compute_ap_range_list(gt_bbox[gt_class_id==i], gt_class_id[gt_class_id==i], gt_mask[:,:,gt_class_id==i], pred_bbox[pred_class_id==i], pred_class_id[pred_class_id==i], pred_scores[pred_class_id==i], pred_mask[:,:,pred_class_id==i], verbose=0)
            mAP = np.array(AP_list).mean()
            AP50 = AP_list[0]
            #print("AP50 fopr category {} is {}".format(i, AP50))
        else:
            mAP = np.nan
            AP50 = np.nan
            AP_list = [np.nan]*10

        #print("mAPlist category {} is: {}".format(i, AP_list))
        #print("gt_masks", gt_mask[:,:,gt_class_id==i].shape[-1])
        #print("predicted masks", pred_mask[:,:,pred_class_id==i].shape[-1])
        #print("AP50 out of if else for category {} is {}".format(i, AP50))
        AP_general.extend([mAP, AP50, AP_list])

    return AP_general, AP_names #mAP_group_values #should be a list of lists of names and values
###########################################################################
# Calculate TP_FP_NF_per_score_mask. General and per class.
###########################################################################
def TP_FP_FN_group(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5):
    TP_FP_FN_general = []
    TP_FP_FN_names = ["score_range", "TP", "FP", "FN","TP_ring", "FP_ring", "FN_ring","TP_crack", "FP_crack", "FN_crack","TP_resin", "FP_resin", "FN_resin", "TP_pith", "FP_pith", "FN_pith"]
    # if no mask is detected
    if pred_mask.shape[-1] == 0:
        score_range = np.arange(0.5, 1.0, 0.05)
        FN = [gt_mask.shape[-1]]*10 # FN here is sum of all the ground truth masks
        TP_FP_FN_general = [score_range, [0]*10, [0]*10, FN]
        for i in range(1,5):
            TP = [0]*10
            FP = [0]*10
            FN = [gt_mask[:,:,gt_class_id==i].shape[-1]]*10
            TP_FP_FN_general.extend([TP, FP, FN])

    else:
        # for all classes
        TP, FP, FN, score_range = TP_FP_FN_per_score_mask(gt_mask, pred_mask, pred_scores, IoU_treshold=IoU_treshold)
        TP_FP_FN_general = [score_range, TP, FP, FN]
        for i in range(1,5):
            TP, FP, FN, score_range = TP_FP_FN_per_score_mask(gt_mask[:,:,gt_class_id==i], pred_mask[:,:,pred_class_id==i], pred_scores[pred_class_id==i], IoU_treshold=IoU_treshold)
            TP_FP_FN_general.extend([TP, FP, FN])

    return TP_FP_FN_general, TP_FP_FN_names

###########################################################################
# Calculate IoU general and per class.
###########################################################################
def IoU_group(gt_mask, gt_class_id, pred_mask, pred_class_id):
    IoU_general = []
    IoU_names = ["IoU", "IoU_ring", "IoU_crack", "IoU_resin", "IoU_pith"]
    # if no mask is detected
    if pred_mask.shape[-1] == 0:
        IoU_general = [0]*5
    else:
        IoU= utils.compute_overlaps_masks(gt_mask, pred_mask)
        IoU = np.nan_to_num(np.mean(IoU)) #change nans to 0
        IoU_general = [IoU]
        for i in range(1,5):
            IoU= utils.compute_overlaps_masks(gt_mask[:,:,gt_class_id==i], pred_mask[:,:,pred_class_id==i])
            IoU = np.nan_to_num(np.mean(IoU)) #change nans to 0
            IoU_general.append(IoU)

    return IoU_general, IoU_names
###########################################################################
# now calculate values for whole dataset
###########################################################################

"""
# variables to calculate
## AP group
mAP = []
AP50 = []
APlist = []

## TP_FP_FN_group
TP = []
FP = []
FN = []

## IoU_group
IoU = []

#COMBINED MASK
IoU_combined_mask = []
TPs_combined = []
FPs_combined = []
FNs_combined = []

# Main structure
for image_id in image_ids:

    ## Load the ground truth for the image
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    print('EVALUATING IMAGE:', image_id)
    imgheight = image.shape[0]

###### Detect image in normal orientation
    results = model.detect([image], verbose=0)
    r = results[0]
    mask_normal = r['masks'] # for the combined mask at the end
    mask_normal_classes = r['class_ids']
    #print("check shapes", gt_class_id, gt_mask[:,:,gt_class_id==1].shape)
    # pass this r to the functions
    #(image, image_meta, gt_class_id, gt_bbox, gt_mask, pred_bbox, pred_mask, pred_class_id, pred_scores)
    AP_general, AP_names = mAP_group(image, gt_class_id, gt_bbox, gt_mask, r['rois'], r['masks'], r['class_ids'], r['scores'])

    mAP.append(AP_general[AP_names.index("mAP")])
    AP50.append(AP_general[AP_names.index("AP50")])
    APlist.append(AP_general[AP_names.index("APlist")])

    #(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5)
    TP_general, TP_names = TP_FP_FN_group(gt_mask, gt_class_id, r['masks'], r['class_ids'], r['scores'], IoU_treshold=0.5)

    TP.append(TP_general[TP_names.index("TP")])
    FP.append(TP_general[TP_names.index("FP")])
    FN.append(TP_general[TP_names.index("FN")])

    IoU_general, IoU_names = IoU_group(gt_mask, gt_class_id, r['masks'], r['class_ids'])

    IoU.append(IoU_general[IoU_names.index("IoU")])


    ## GET COMBINED MASK WITH POSTPROCESSING AND CLACULATE precission, recall and IoU
    #normal flatten
    detected_mask = sliding_window_detection(image = image, modelRing=model, overlap = 0.75, cropUpandDown = 0) # may be without crop
    detected_mask_rings = detected_mask[:,:,0]
    print("detected_mask_rings", detected_mask_rings.shape)
    #print("detected_mask_cracks", detected_mask_cracks.shape)
    clean_contours_rings = clean_up_mask(detected_mask_rings, is_ring=True)
    print("clean_contours_rings", len(clean_contours_rings))

    combined_mask_binary = contours_to_binary(clean_contours_rings, imgheight, imgheight, debug=False)
    print('combined_mask_binary.shape', combined_mask_binary.shape)

    # Ploting lines is moslty for debugging
    file_name = 'image'+ str(image_id)
    masked_image = image.astype(np.uint32).copy()
    masked_image = apply_mask(masked_image, detected_mask_rings, alpha=0.3)
    plot_lines(image=masked_image,file_name=file_name,
                path_out=IMAGE_PATH_OUT, gt_masks=gt_mask[:,:,gt_class_id==1],
                clean_contours = clean_contours_rings, debug=False)

    IoU_combined_mask.append(utils.compute_overlaps_masks(gt_mask_flat_binary, combined_mask_binary_wrong))

    mask_matrix = utils.compute_overlaps_masks(gt_mask, combined_mask_binary)
    print("mask_matrix.shape", mask_matrix.shape)
    print("mask_matrix", mask_matrix)
    # making binary numpy array with IoU treshold
    ## HERE YOU CAN CALCULATE FOR ALL IoU
    IoU_treshold = 0.5 # i set it less because combined mask is bigger and it does not matter i think
    mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
    #print (mask_matrix_binary)


    #GT rings and predicted rigs
    gt_r = len(mask_matrix)
    pred_r = len(mask_matrix[0])

    #TP
    sum_truth = np.sum(mask_matrix_binary, axis=1)
    sum_truth_binary = np.where(sum_truth > 0, 1, 0)
    TP_comb = np.sum(sum_truth_binary)
    TPs_combined.append(TP_comb)
    #print('TP:', TP)
    #FP
    sum_pred = np.sum(mask_matrix_binary, axis=0)
    sum_pred_binary = np.where(sum_pred > 0, 1, 0)
    FP_comb = pred_r - np.sum(sum_pred_binary)
    FPs_combined.append(FP_comb)
    #print('FP:', FP)
    #FN
    FN_comb = gt_r - TP_comb
    FNs_combined.append(FN_comb)
#print("IoU_combined_mask",IoU_combined_mask)

#calculate averages for all images
#0
## AP group
mAP = np.nanmean(mAP)
print("mAP", mAP)
AP50 = np.nanmean(AP50)
APlist = np.nanmean(APlist, axis=0)

## TP_FP_FN_group
TP = np.array(np.sum(TP, axis=0))
FP = np.array(np.sum(FP, axis=0))
FN = np.array(np.sum(FN, axis=0))
### calculate sensitivity and precission
SEN = TP/(TP+FN)
PREC = TP/(TP+FP)
#print("SEN", SEN)
#print("PREC", PREC)

## IoU_group
IoU = np.mean(IoU)

# Prec and recall for combined
#print('TPs_combined', TPs_combined)
TPs_combined = np.sum(TPs_combined)
FPs_combined = np.sum(FPs_combined)
FNs_combined = np.sum(FNs_combined)
#print('TPs_combined', TPs_combined)
SEN_combined = TPs_combined/(TPs_combined+FNs_combined)
PREC_combined = TPs_combined/(TPs_combined+FPs_combined)
IoU_combined_mask = np.mean(IoU_combined_mask)

iou_thresholds = np.arange(0.5, 1.0, 0.05) # for graph

#######################################################################
#Save output
#######################################################################
#get folder path and make folder
weight_path = args.weight
#print(run_path)
weight_path_split_1 = os.path.split(weight_path)
#print(run_split_1)
weight_name = weight_path_split_1[1]
#print('weight_name:', weight_name)
training_ID = os.path.split(weight_path_split_1[0])[1]
#print('run_ID:', run_ID)

model_eval_DIR = os.path.join(ROOT_DIR, 'samples/TreeRing/model_eval')
#print(model_eval_DIR)
training_eval_DIR = os.path.join(model_eval_DIR,training_ID)
weight_eval_DIR = os.path.join(training_eval_DIR, weight_name, args.TreeRingConf)

if not os.path.exists(training_eval_DIR): #check if it already exists and if not make it
    os.makedirs(training_eval_DIR)

if not os.path.exists(weight_eval_DIR): #check if it already exists and if not make it
    os.makedirs(weight_eval_DIR)

#save table
df = pd.DataFrame()

df['variables'] = ["mAP", "AP50", "TP", "FP", "FN", "SEN", "PREC", "IoU"] #names of all the variables

df['values'] = [mAP, AP50, TP[0], FP[0], FN[0], SEN[0], PREC[0], IoU]#values for all the variables

# save the df as csv
df.to_csv(os.path.join(weight_eval_DIR, 'Evaluation_{}.csv'.format(weight_name)))

# save mAP graph
plt.plot(iou_thresholds, APlist, label= 'normal')
plt.ylabel('mAP')
plt.xlabel('IoU')
plt.legend()
#plt.ylim(bottom = 0, top = 1)
#plt.xlim(left = 0, right = 1)
plt.savefig(os.path.join(weight_eval_DIR, 'mAP_IoU_{}.jpg'.format(weight_name)))
plt.close()

##save graph data
df_mAP_graph = pd.DataFrame()
df_mAP_graph['mAP'] = APlist
df_mAP_graph['IoU_thresholds'] = iou_thresholds
df_mAP_graph.to_csv(os.path.join(weight_eval_DIR, 'mAP_IoU_graph_data_{}.csv'.format(weight_name)))

#save precission recall graph for mask and box together
plt.plot(SEN, PREC, label= 'general')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim(bottom = 0, top = 1)
plt.xlim(left = 0, right = 1)
plt.legend()
plt.savefig(os.path.join(weight_eval_DIR, 'PrecRec_{}.jpg'.format(weight_name)))
plt.close()

##save precission recall data
df_PrecRec = pd.DataFrame()
df_PrecRec['Prec'] = PREC
df_PrecRec['Rec'] = SEN

df_PrecRec.to_csv(os.path.join(weight_eval_DIR, 'PrecRec_graph_data_{}.csv'.format(weight_name)))
"""
"""
######## TEST EVALUATE_TRAINING in DEVELOPMENT STAGE
training_log = "/Users/miroslav.polacek/Github/TRG_development/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/logs/fake_training"
dataset = "/Users/miroslav.polacek/Github/TRG_development/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/training/sample_dataset"
dataset = "/Users/miroslav.polacek/Documents/DebugDataset_temp"
test_output = evaluate_training(model, training_log, dataset)
"""

def evaluate_training(model, training_log, dataset):
    # Validation dataset
    ## Later can be extyended to take any folder with properly annotated images, for now it has to be in subfolder val
    dataset_val = TreeRing_onlyRing.TreeringDataset()
    dataset_val.load_treering(dataset, "val")
    dataset_val.prepare()

    # list weights names, .h5 to check
    weight_list_raw = os.listdir(training_log)
    weight_list = []
    for file_name in weight_list_raw:
        if file_name.endswith(".h5"):
            weight_list.append(file_name)
        else:
            print(f"File {file_name} is not valid weight")
    print("List of weight to evaluate", weight_list)
    # prepare panda dataframe to append
    df_out = pd.DataFrame(columns=["weight_name", "mAP", "AP50", "TP", "FP", "FN", "SEN", "PREC", "IoU",
                                   "TP_com", "FP_com", "FN_com", "SEN_com", "PREC_com"])
    # look through weights
    for weight_name in weight_list:
        weight_path = os.path.join(training_log, weight_name)
        # Load weights
        print("Loading weights from", weight_path)
        model.load_weights(weight_path, by_name=True)
        image_ids = dataset_val.image_ids

        ##### evaluate on val set, export line of values to append to a dataframe #####
        # variables to calculate
        mAP = []
        AP50 = []
        APlist = []
        TP = []
        FP = []
        FN = []
        IoU = []

        # COMBINED MASK
        IoU_combined_mask = []
        TPs_combined = []
        FPs_combined = []
        FNs_combined = []

        # Main structure
        for image_id in image_ids:
            ## Load the ground truth for the image
            image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                modellib.load_image_gt(dataset_val, config,
                                       image_id, use_mini_mask=False)
            print('EVALUATING IMAGE:', image_id)
            imgheight = image.shape[0]

            ###### Detect image in normal orientation
            results = model.detect([image], verbose=0)
            r = results[0]
            mask_normal = r['masks']  # for the combined mask at the end
            mask_normal_classes = r['class_ids']
            # print("check shapes", gt_class_id, gt_mask[:,:,gt_class_id==1].shape)
            # pass this r to the functions
            # (image, image_meta, gt_class_id, gt_bbox, gt_mask, pred_bbox, pred_mask, pred_class_id, pred_scores)
            AP_general, AP_names = mAP_group(image, gt_class_id, gt_bbox, gt_mask, r['rois'], r['masks'],
                                             r['class_ids'], r['scores'])

            mAP.append(AP_general[AP_names.index("mAP")])
            AP50.append(AP_general[AP_names.index("AP50")])
            APlist.append(AP_general[AP_names.index("APlist")])

            # (gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5)
            TP_general, TP_names = TP_FP_FN_group(gt_mask, gt_class_id, r['masks'], r['class_ids'], r['scores'],
                                                  IoU_treshold=0.5)

            TP.append(TP_general[TP_names.index("TP")])
            FP.append(TP_general[TP_names.index("FP")])
            FN.append(TP_general[TP_names.index("FN")])

            IoU_general, IoU_names = IoU_group(gt_mask, gt_class_id, r['masks'], r['class_ids'])

            IoU.append(IoU_general[IoU_names.index("IoU")])

            ## GET COMBINED MASK WITH POSTPROCESSING AND CLACULATE precission, recall and IoU
            # may be without cropUpandDown
            detected_mask = sliding_window_detection_multirow(image=image,
                                              detection_rows=1,
                                              modelRing=model,
                                              overlap=0.75,
                                              cropUpandDown=0)
            detected_mask_rings = detected_mask[:, :, 0]
            #print("detected_mask_rings", detected_mask_rings.shape)
            # print("detected_mask_cracks", detected_mask_cracks.shape)
            clean_contours_rings = clean_up_mask(detected_mask_rings, is_ring=True)
            #print("clean_contours_rings", len(clean_contours_rings))

            combined_mask_binary = contours_to_binary(clean_contours_rings, imgheight, imgheight, debug=False)
            #print('combined_mask_binary.shape', combined_mask_binary.shape)

            gt_mask_flat = gt_mask.sum(axis=2)
            #print("gt_mask_flat.shape", gt_mask_flat.shape)
            combined_mask_binary_flat = combined_mask_binary.sum(axis=2)
            #print("combined_mask_binary_flat.shape", combined_mask_binary_flat.shape)

            mask_matrix = utils.compute_overlaps_masks(gt_mask, combined_mask_binary)

            # making binary numpy array with IoU treshold
            ## HERE YOU CAN CALCULATE FOR ALL IoU
            IoU_treshold = 0.5  # i set it less because combined mask is bigger and it does not matter i think
            mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
            # print (mask_matrix_binary)

            # GT rings and predicted rigs
            gt_r = len(mask_matrix)
            pred_r = len(mask_matrix[0])

            # TP
            sum_truth = np.sum(mask_matrix_binary, axis=1)
            sum_truth_binary = np.where(sum_truth > 0, 1, 0)
            TP_comb = np.sum(sum_truth_binary)
            TPs_combined.append(TP_comb)
            # print('TP:', TP)
            # FP
            sum_pred = np.sum(mask_matrix_binary, axis=0)
            sum_pred_binary = np.where(sum_pred > 0, 1, 0)
            FP_comb = pred_r - np.sum(sum_pred_binary)
            FPs_combined.append(FP_comb)
            # print('FP:', FP)
            # FN
            FN_comb = gt_r - TP_comb
            FNs_combined.append(FN_comb)
        # calculate averages for all images
        # 0
        ## AP group
        mAP = np.nanmean(mAP)
        print("mAP", mAP)
        AP50 = np.nanmean(AP50)

        ## TP_FP_FN_group
        TP = np.array(np.sum(TP, axis=0))
        FP = np.array(np.sum(FP, axis=0))
        FN = np.array(np.sum(FN, axis=0))
        ### calculate sensitivity and precission
        SEN = TP / (TP + FN)
        PREC = TP / (TP + FP)
        # print("SEN", SEN)
        # print("PREC", PREC)

        ## IoU_group
        IoU = np.mean(IoU)

        # Prec and recall for combined
        # print('TPs_combined', TPs_combined)
        TPs_combined = np.sum(TPs_combined)
        FPs_combined = np.sum(FPs_combined)
        FNs_combined = np.sum(FNs_combined)
        # print('TPs_combined', TPs_combined)
        SEN_combined = TPs_combined / (TPs_combined + FNs_combined)
        PREC_combined = TPs_combined / (TPs_combined + FPs_combined)


        # append row to a dataframe, column names are "weight_name", "mAP", "AP50", "TP", "FP", "FN", "SEN", "PREC", "IoU,
        #                                    "TP_com", "FP_com", "FN_com", "SEN_com", "PREC_com"
        df_out.loc[len(df_out.index)] = [weight_name, mAP, AP50, TP[0], FP[0], FN[0], SEN[0], PREC[0], IoU,
                                         TPs_combined, FPs_combined, FNs_combined, SEN_combined,
                                         PREC_combined]
        df_out.sort_values(by="mAP", ascending=False, inplace=True)
        # save the csv in the training log folder
        df_out.to_csv(os.path.join(training_log, "weights_stats.csv"))

    return df_out

def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_logs', required=False)

    parser.add_argument('--dataset', required=False)

    args = parser.parse_args()
    return args
print("got almost to the end")

def main():
    print("got in the main")
    args = get_args()
    training_log = args.train_logs
    dataset = args.dataset
    _ = evaluate_training(model, training_log, dataset)

if __name__ == '__main__':
    main()
