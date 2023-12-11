import pandas as pd
import gc
import os
import json

##########################
# global variables to change:
MODE = "train"
# MODE = "validation"
##########################

# it is assumed that previous steps were completed and partial files already generated
# all files were generated either with BPP True or with BPP False in previous step

# the script needs to be run twice: once for training and once for validation data (change global variables accordingly)
# result: ready-to-use training and validation files will be generated
if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        data = json.load(f)
    if MODE == "train":
        dir_to_read = data["PARTIAL_FILES_TRAIN_DIR"]
        path_to_write = data["TRAIN_DATA"]
    else:
        dir_to_read = data["PARTIAL_FILES_VAL_DIR"]
        path_to_write = data["VAL_DATA"]

    file_0 = os.path.join(dir_to_read, "0.parquet")
    file_1 = os.path.join(dir_to_read, "1.parquet")

    dfs = []
    df_1 = pd.read_parquet(file_0, engine='pyarrow')
    dfs.append(df_1)
    df_2 = pd.read_parquet(file_1, engine='pyarrow')
    dfs.append(df_2)
    df_1 = pd.concat(dfs, ignore_index=True)
    del dfs, df_2
    gc.collect()
    print("2 files collected")

    if MODE == "train":
        end_number = 16
    else:
        end_number = 3
    for i in range(2, end_number + 1):
        next_file = os.path.join(dir_to_read, f"{i}.parquet")
        dfs = []
        dfs.append(df_1)
        df_2 = pd.read_parquet(next_file, engine='pyarrow')
        dfs.append(df_2)
        df_1 = pd.concat(dfs, ignore_index=True)
        del dfs, df_2
        gc.collect()
        print(f"{i+1} files collected")

    # row_group_size is necessary to avoid mistake when this file will be read (with bpp information)
    # https://www.vortexa.com/insights/technology/when-parquet-columns-get-too-big/
    df_1.to_parquet(path_to_write, engine='pyarrow', index=False, row_group_size=10000)

    print(f"length of resulting df is: {len(df_1)}.")
