import os
import pandas as pd
import numpy as np

####### Functions #####################
folder_path=LOGS_2
def get_speed_df(folder_path):
    log_file_gen = (f for f in os.listdir(folder_path) if f.endswith('.log'))
    im_name_list = []
    time_list = []
    rings_list = []
    for log_file in log_file_gen:
        with open(os.path.join(folder_path, log_file), 'r') as f:
            for line in f:
                if "Processing image:" in line:
                    im_name = line.split("Processing image: ")[1].split("\n")[0]
                    im_name_list.append(im_name)
                if "Image run time: " in line:
                    time_s = float(line.split("Image run time: ")[1].split(" s\n")[0])
                    time_list.append(time_s)
                if "Filtered_centerlines: " in line:
                    rings = int(line.split("Filtered_centerlines: ")[1].split("\n")[0])
                    rings_list.append(rings)
    if len(rings_list) == 0:
        speed_df = pd.DataFrame({"im_name": im_name_list, "time_s": time_list})
    else:
        speed_df = pd.DataFrame({"im_name": im_name_list, "time_s": time_list, "rings": rings_list})

    return speed_df
#######################################

LOGS_1 = "/Users/miroslav/Documents/SpeedTest_HPC2N_no_debug_pygeoops"
LOGS_2 = "/Users/miroslav/Documents/SpeedTest_HPC2N_no_debug_ultra83P3123test_CV411Cuda124"

df_1 = get_speed_df(LOGS_1)
df_2 = get_speed_df(LOGS_2)

### NO DEBUG ####
mean1 = np.mean(df_1["time_s"])/60
min1 = np.min(df_1["time_s"])
max1 = np.max(df_1["time_s"])
rings_sum = np.sum(df_1["rings"])

### WITH DEBUG ##### in minutes
mean2 = np.mean(df_2["time_s"])/60
min2 = np.min(df_2["time_s"])/60
max2 = np.max(df_2["time_s"])/60

dif = abs(mean1 - mean2)
prop_dif = dif/np.max([mean1, mean2])