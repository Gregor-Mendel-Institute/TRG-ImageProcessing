import os
import pandas as pd

####### Functions #####################
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

LOGS_NO_Debug_NO_Png = "/Volumes/swarts/user/miroslav.polacek/Container_test/output/speed_test_no_debug_and_png"
LOGS_Debug_Png = "/Volumes/swarts/user/miroslav.polacek/Container_test/output/speed_test_debug_and_png"

df_New_No_debug_No_png = get_speed_df(LOGS_NO_Debug_NO_Png)
df_New_debug_png = get_speed_df(LOGS_Debug_Png)




