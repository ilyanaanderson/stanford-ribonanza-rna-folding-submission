import os
from parallel_pandas import ParallelPandas
import json
import pandas as pd

from data_utils import all_info_from_seq, all_info_from_seq_no_bpp

##########################
# global variables to change:

MODE = "train"
# MODE = "validation"

BPP = False
##########################

# if BPP is True, it extracts bpp information in addition to everything else
# this will require a computer with 128 GB of memory in subsequent steps

# set BPP either to True or to False
# the script will have to be re-run two times
# run one time with MODE = "train" and one time with MODE = "validation"

# the result: small data files will be created (10k rows or less)
if __name__ == '__main__':
    ParallelPandas.initialize(n_cpu=None, split_factor=4, disable_pr_bar=False)

    with open('SETTINGS.json') as f:
        data = json.load(f)
    if MODE == "train":
        path_to_read = data["INTERMEDIATE_TRAIN_DATA"]
        dir_to_write = data["PARTIAL_FILES_TRAIN_DIR"]
        last_file_num = 16
    else:
        path_to_read = data["INTERMEDIATE_VAL_DATA"]
        dir_to_write = data["PARTIAL_FILES_VAL_DIR"]
        last_file_num = 3

    df = pd.read_parquet(path_to_read, engine='pyarrow')
    step = 10000

    if BPP:
        func = all_info_from_seq
    else:
        func = all_info_from_seq_no_bpp

    for i in range(last_file_num):
        path_to_write = os.path.join(dir_to_write, f"{i}.parquet")
        idx_begin = i * step
        idx_end = idx_begin + step
        df_tmp = df[idx_begin:idx_end]
        final_df = df_tmp.p_apply(func, axis=1)
        final_df.to_parquet(path_to_write, index=False, engine='pyarrow')
    # last file
    path_to_write = os.path.join(dir_to_write, f"{last_file_num}.parquet")
    idx_begin = last_file_num * step
    df_tmp = df[idx_begin:]
    final_df = df_tmp.p_apply(func, axis=1)
    final_df.to_parquet(path_to_write, index=False, engine='pyarrow')

