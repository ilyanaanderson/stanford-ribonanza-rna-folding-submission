from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
from arnie.mfe import mfe
from arnie.bpps import bpps


TARGET_LEN = 457
TARGET_LEN_EOS = 459
NUM_OF_REACTIVITIES = 206
EOS = 2
BOS = 1
NUCLEOTIDES_DICT = {
    'A': 3,
    'C': 4,
    'G': 5,
    'U': 6
}
NUCLEOTIDES_STRUCT_DICT = {
    'A(': 7,
    'A.': 8,
    'A)': 9,
    'C(': 10,
    'C.': 11,
    'C)': 12,
    'G(': 13,
    'G.': 14,
    'G)': 15,
    'U(': 16,
    'U.': 17,
    'U)': 18,
}
#######################################
# new global vars and dicts
STRUCT_DICT = {
    '.': 3,
    '(': 4,
    ')': 5,
}
NUCLEOTIDES_STRUCT_DICT_NEW = {
    'A(': 3,
    'A.': 4,
    'A)': 5,
    'C(': 6,
    'C.': 7,
    'C)': 8,
    'G(': 9,
    'G.': 10,
    'G)': 11,
    'U(': 12,
    'U.': 13,
    'U)': 14,
}


#######################################
# the processing of data happens in two stages
# stage one is processing related to reactivities
# stage two is working with sequence

##########################################
# stage one functions:
def noclip_nofilter_all_info(row):
    reactivity_data = row.filter(like='reactivity_0')
    errors_data = row.filter(like='reactivity_error')
    nump = reactivity_data.to_numpy(dtype=np.float32)
    error = errors_data.to_numpy(dtype=np.float32)
    non_nan_mask = ~np.isnan(nump)
    non_nan_count = np.sum(non_nan_mask)
    return pd.Series([row['sequence'], nump, error, non_nan_count],
                     index=['sequence', 'nump_react', 'error', 'non_nan_count'])


def noclip_filter03_all_info(row):
    reactivity_data = row.filter(like='reactivity_0')
    errors_data = row.filter(like='reactivity_error')
    nump = reactivity_data.to_numpy(dtype=np.float32)
    nump_err = errors_data.to_numpy(dtype=np.float32)
    # will use reactivity errors to turn some nump values into nan
    mask = (~np.isnan(nump_err)) & (nump_err > 0.3)  # true if error is not nan (is float) and too large
    nump[mask] = np.nan

    non_nan_mask = ~np.isnan(nump)
    non_nan_count = np.sum(non_nan_mask)
    return pd.Series([row['sequence'], nump, nump_err, non_nan_count],
                     index=['sequence', 'nump_react', 'error', 'non_nan_count'])

