import os
from parallel_pandas import ParallelPandas
import json
import pandas as pd

from data_utils import all_info_from_seq, all_info_from_seq_no_bpp

##########################
# global variables to change:

MODE = "train"
# MODE = "validation"

FILE_NUMBER = 0    # goes from 0 to 16 (inclusive) for train, and 0 to 3 (inclusive) for validation
# FILE_NUMBER is ignored if BPP is False

BPP = False
##########################

# if BPP is True, it extracts bpp information in addition to everything else
# this script will have to be manually re-run 17 times (for train) and 4 times (for validation),
# with minor changes to this script: change global variables above (MODE; FILE_NUMBER; set BPP to True)
# this is because calculating bpps is computationally expensive and require a lot of memory
# if all files are created in one run, it might slow down or crash
# problematic if more than 10k rows are processed at a time

# if BPP is False, the script will have to be re-run two times
# run one time with MODE = "train" and one time with MODE = "validation"

# the result: small data files will be created (10k rows or less)
if __name__ == '__main__':
    ParallelPandas.initialize(n_cpu=None, split_factor=4, disable_pr_bar=False)

    with open('SETTINGS.json') as f:
        data = json.load(f)
    if MODE == "train":
        path_to_read = data["INTERMEDIATE_TRAIN_DATA"]
        dir_to_write = data["PARTIAL_FILES_TRAIN_DIR"]
    else:
        path_to_read = data["INTERMEDIATE_VAL_DATA"]
        dir_to_write = data["PARTIAL_FILES_VAL_DIR"]

    df = pd.read_parquet(path_to_read, engine='pyarrow')
    step = 10000

    if BPP:
        path_to_write = os.path.join(dir_to_write, f"{FILE_NUMBER}.parquet")
        idx_begin = FILE_NUMBER * step
        idx_end = idx_begin + step

        if MODE == "train" and FILE_NUMBER == 16:
            end = True
        elif MODE == "validation" and FILE_NUMBER == 3:
            end = True
        else:
            end = False

        if not end:
            df = df[idx_begin:idx_end]
        else:
            df = df[idx_begin:]

        final_df = df.p_apply(all_info_from_seq, axis=1)
        final_df.to_parquet(path_to_write, index=False, engine='pyarrow')
    else:
        # bpp information is not collected
        # runs faster and files are smaller: for-loop is used to create multiple files
        # possible to re-write this script to process the entirety of df at once
        # no partial files will be needed in that case
        # however, this step is set up in a way to accommodate both ways of extracting information,
        # with bpp and without (so partial files are created for both ways)
        if MODE == "train":
            last_file_num = 16
        else:
            last_file_num = 3
        for i in range(last_file_num):
            path_to_write = os.path.join(dir_to_write, f"{i}.parquet")
            idx_begin = i * step
            idx_end = idx_begin + step
            df_tmp = df[idx_begin:idx_end]
            final_df = df_tmp.p_apply(all_info_from_seq_no_bpp, axis=1)
            final_df.to_parquet(path_to_write, index=False, engine='pyarrow')
        # last file
        path_to_write = os.path.join(dir_to_write, f"{last_file_num}.parquet")
        idx_begin = last_file_num * step
        df_tmp = df[idx_begin:]
        final_df = df_tmp.p_apply(all_info_from_seq_no_bpp, axis=1)
        final_df.to_parquet(path_to_write, index=False, engine='pyarrow')

