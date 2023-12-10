import os
from parallel_pandas import ParallelPandas
import json
import pandas as pd

from data_utils import all_info_from_seq

##########################
# global variables to change:
MODE = "train"
# MODE = "validation"
FILE_NUMBER = 15    # goes from 0 to 16 for train, and 0 to 3 for validation
##########################

# this script will have to be re-run 17 times (for train) and 4 times (for valid), with minor change to this script:
# change global variables above
# the result: small data files will be created (10k rows or less)
# it is possible to create all these files in one run, but it might be better to let the computer rest between runs
# if all files are created in one run, it might slow down or crash
# problematic if more than 10k rows are processed at a time
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

    path_to_write = os.path.join(dir_to_write, f"{FILE_NUMBER}.parquet")
    step = 10000
    idx_begin = FILE_NUMBER * step
    idx_end = idx_begin + step

    if MODE == "train" and FILE_NUMBER == 16:
        end = True
    elif MODE == "validation" and FILE_NUMBER == 3:
        end = True
    else:
        end = False

    df = pd.read_parquet(path_to_read, engine='pyarrow')
    if not end:
        df = df[idx_begin:idx_end]
    else:
        df = df[idx_begin:]

    final_df = df.p_apply(all_info_from_seq, axis=1)

    final_df.to_parquet(path_to_write, index=False, engine='pyarrow')

