import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
####### Args ##########################
def get_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Segmentation of whole core')

    ## Compulsory arguments
    parser.add_argument('--log_folder', required=True,
                        help="Folder containing log file")

    parser.add_argument('--out_folder', required=False,
                        default=None,
                        help="Folder where to save results. If not specified log_folder will be used")
    args = parser.parse_args()
    return args
####### Functions #####################
# folder_path=LOGS_1
def get_speed_df(log_file_path):
    item_list = []
    item = []
    gpu_info = []
    with open(log_file_path, 'r') as f:
        for line in f:
            if "Model is running on: " in line:
                gpu = line.split("Model is running on: ")[1].split("\n")[0]
                gpu_info.append(gpu)
            if "Processing image:" in line:
                # here I relay on a structure of the log file assuming the image is the first thing saved per core
                item_list.append(item)
                item = []
                im_name = line.split("Processing image: ")[1].split("\n")[0]
                #im_name_list.append(im_name)
                item.append(im_name)
            if "Image has height" in line:
                height = int(line.split("Image has height ")[1].split(" and width ")[0])
                width = int(line.split(" and width ")[1].split("\n")[0])
                item.append(height)
                item.append(width)
            if "Filtered_centerlines:" in line:
                rings = int(line.split("Filtered_centerlines: ")[1].split("\n")[0])
                #rings_list.append(rings)
                item.append(rings)
            if "IMAGE WAS NOT FINISHED" in line:
                item.append(0)
            elif "IMAGE FINISHED" in line:
                item.append(1)
            if "Image run time:" in line:
                time_s = float(line.split("Image run time: ")[1].split(" s\n")[0])
                #time_list.append(time_s)
                item.append(time_s)

    cleaned_list = []
    # here it should contain [name, rings, status, time]
    for i in item_list:
        if len(i) == 6:
            cleaned_list.append(i)
        elif len(i) == 5:
            i.append(pd.NA) # add NA for missing time as a placeholder
            cleaned_list.append(i)
        else:
            print(f"Item {i} removed in cleaning")
    try:
        speed_df = pd.DataFrame(data=np.array(cleaned_list), columns=["im_name", "rings", "im_height", "im_width", "status", "time_s"])
        speed_df[["im_height", "im_width", "rings", "status"]] = speed_df[["im_height", "im_width", "rings", "status"]].astype(int)
        speed_df[["time_s"]] = speed_df[["time_s"]].astype(float)
    except Exception as e:
        print(f"Data type conversion failed: {e}")

    return speed_df, gpu_info

def get_speed_df_from_folder(log_folder_path):
    log_file_gen = (f for f in os.listdir(log_folder_path) if f.endswith('.log'))
    speed_df_list = []
    gpu_info_list = []
    for log_file in log_file_gen:
        speed_df, gpu_info  = get_speed_df(os.path.join(log_folder_path, log_file))
        speed_df_list.append(speed_df)
        gpu_info_list.append(gpu_info)

    speed_df_complete = pd.concat(speed_df_list, ignore_index=True)

    return speed_df_complete, gpu_info_list

def scatterplot_speeds(speed_df, out_path):
    fig = sns.scatterplot(speed_df, x="size_ratio", y="time_m", size="rings").get_figure()
    #fig = sns.lmplot(speed_df, x="size_ratio", y="time_m", size="rings").get_figure()
    #fig.show()
    fig.savefig(os.path.join(out_path, "Processing_speed.png"), dpi=300, format='png')
    #save plot png

def speed_log_eval(log_folder_path, out_path=None):
    sdf, gpu = get_speed_df_from_folder(log_folder_path)
    # set up output path
    if out_path is None:
        out_path = log_folder_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    # derive additional variables
    sdf["time_m"] = sdf["time_s"] / 60
    sdf["size_ratio"] = sdf["im_width"] / sdf["im_height"]
    # save raw speed df as table
    sdf.to_csv(os.path.join(out_path, "speed_rings_per_core.csv"), index=False)
    # clean unfinished images
    sdf_clean = sdf[sdf['status'] == 1]
    # save stats from cleaned
    ## one row of table that can have gpu, average speed, min, max, n_samples
    mean, min, max, n = sdf_clean["time_m"].mean(), sdf_clean["time_m"].min(), sdf_clean["time_m"].max(), sdf_clean["time_m"].count()
    #speed_summary_df = pd.DataFrame([mean, min, max, n, gpu], columns=["mean", "min", "max", "count", "device"])
    speed_summary_df = pd.DataFrame({"mean": mean, "min": min, "max": max, "n": n, "device": gpu}, columns=["mean", "min", "max", "n", "device"])
    speed_summary_df.to_csv(os.path.join(out_path, "speed_summary.csv"), index=False)
    # plot the cleaned data
    scatterplot_speeds(sdf_clean, out_path)

#######################################
def main():
    args = get_args()
    if not os.path.exists(args.log_folder):
        print(f"Log folder does not exist: {args.log_folder}")

    print("Output path: ", args.out_folder)
    speed_log_eval(args.log_folder, out_path=args.out_folder)

if __name__ == "__main__":
    main()