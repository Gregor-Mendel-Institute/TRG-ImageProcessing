import os
import pandas as pd
import numpy as np

#### there was no speed in the old log files so I have to generate newer with old version to be able to compare this
## This code will be usefull pater
JSON_FOLDER_DIR_old = '/Volumes/swarts/lab/ImageProcessingPipeline/BashScripts/oldRingWidthsFromSwartsShiny/oldRingWidths/19/20240925'
JSON_FOLDER_DIR_new = '/Volumes/swarts/user/miroslav.polacek/Container_test/output/19'
# to test f_dir=JSON_FOLDER_DIR_old
json_size_df = pd.DataFrame(columns=["json_mame", "size", "new"])
for i, f_dir in enumerate((JSON_FOLDER_DIR_old, JSON_FOLDER_DIR_new)):
    for json in (j for j in os.listdir(f_dir) if j.endswith('.json')):
        print(json)
        json_size = round(os.path.getsize(os.path.join(f_dir, json))/(pow(1024,2)), 2)
        print(f"{json_size} MB")
        df_concat = pd.DataFrame({"json_mame": json, "size": json_size, "new": i}, index=[0])
        json_size_df = pd.concat([json_size_df,df_concat ], ignore_index=True) #{"json_mame": json, "size": json_size}

# pivot to columns
dfp = json_size_df.pivot(index="json_mame", columns="new", values="size")

# get only data for images that were in both tables
dfp_clean = dfp.dropna()
mean_old = np.mean(dfp_clean[0].tolist())
std_old = np.std(dfp_clean[0].tolist())
mean_new = np.mean(dfp_clean[1].tolist())
std_new = np.std(dfp_clean[1].tolist())






# for old and new
# list of jsons in there
# for json in folder
##get size
# os.path.getsize('flickrapi-1.2.tar.gz')
## save the name and the size in pandas dataframe

# merge by name with old and new size as columns
