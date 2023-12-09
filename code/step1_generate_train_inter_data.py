import pandas as pd
import gc
from parallel_pandas import ParallelPandas
import json

from data_utils import noclip_nofilter_all_info


WRITE_RAW = '../data/train_proc/raw/SN_noclip_nofilter.parquet'

# task: take Quick_start_data and create parquet intermediate training file
# this file will be later on processed further
# since the Quick_start_data is clean, not all steps are necessary
# (e.g., dropping duplicates is not needed, but it is fast)
# similar script will be run for data, which is not clean, later on
if __name__ == '__main__':
    ParallelPandas.initialize(n_cpu=None, split_factor=1, disable_pr_bar=False)

    # data will be read using the path from json file
    with open('SETTINGS.json') as f:
        data = json.load(f)
    quick_data = data["RAW_QUICK_DATA_TRAIN"]
    write_intermediate_path = data["INTERMEDIATE_TRAIN_DATA"]

    df = pd.read_csv(quick_data)

    df_2A3 = df.loc[df.experiment_type == '2A3_MaP']
    df_DMS = df.loc[df.experiment_type == 'DMS_MaP']

    del df
    gc.collect()

    len_a = len(df_2A3)
    len_d = len(df_DMS)
    print(f"len of a is {len_a}, len of d is {len_d}")

    # make it unique sequences only:
    df_2A3.drop_duplicates(subset='sequence', keep='first', inplace=True)
    df_DMS.drop_duplicates(subset='sequence', keep='first', inplace=True)

    len_a = len(df_2A3)
    len_d = len(df_DMS)
    print(f"after dropping duplicates len of a is {len_a}, len of d is {len_d}")

    func = noclip_nofilter_all_info
    df_2A3 = df_2A3.p_apply(func, axis=1)
    df_DMS = df_DMS.p_apply(func, axis=1)

    df_2A3 = df_2A3[df_2A3['non_nan_count'] != 0]
    df_DMS = df_DMS[df_DMS['non_nan_count'] != 0]

    len_a = len(df_2A3)
    len_d = len(df_DMS)
    print(f"after dropping rows with nan len of a is {len_a}, len of d is {len_d}")

    # change columns:
    dict_for_a = {
        'sequence': 'sequence',
        'nump_react': 'nump_react_a',
        'error': 'error_a',
        'non_nan_count': 'non_nan_count_a'
    }
    dict_for_d = {
        'sequence': 'sequence',
        'nump_react': 'nump_react_d',
        'error': 'error_d',
        'non_nan_count': 'non_nan_count_d'
    }

    new_column_names_a = dict_for_a
    new_column_names_d = dict_for_d
    df_2A3.rename(columns=new_column_names_a, inplace=True)
    df_DMS.rename(columns=new_column_names_d, inplace=True)
    print(df_DMS.columns)

    df = pd.merge(df_2A3, df_DMS, on='sequence', how='inner')

    print("merged")
    df.reset_index(inplace=True, drop=True)
    print("columns:")
    print(df.columns)

    len_df = len(df)
    print(f"length of merged df is {len_df}")

    # write to intermediate:
    df.to_parquet(write_intermediate_path, index=False, engine='pyarrow')
    print("created intermediate train data")




