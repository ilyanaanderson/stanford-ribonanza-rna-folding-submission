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
STRUCT_DICT = {
    '.': 3,
    '(': 4,
    ')': 5,
}
NUCLEOTIDES_STRUCT_DICT = {
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
# the processing of training and validation data happens in two stages
# stage one is processing related to reactivities
# stage two is working with sequence (extracting information from it)
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


################################################
# stage two functions:

# works only on small data (used in this repository for 10k rows)
def all_info_from_seq(row):
    # received 'sequence', 'nump_react_a', 'nump_react_d', 'error_a', 'error_d', 'non_nan_count_a', 'non_nan_count_d'
    seq = row['sequence'].strip()
    s_len = len(seq)
    struct = mfe(seq, package="eternafold")
    # sending bpp (2-D numpy array) is necessary because pyarrow needs something that it can convert to 1-D numpy array
    # pyarrow cannot process value that is 2-D array
    bpp = bpps(seq, package="eternafold").tolist()

    seq_list = [*seq]
    seq_inds = np.array([NUCLEOTIDES_DICT[char] for char in seq_list])
    seq_inds = np.insert(seq_inds, 0, EOS)
    seq_inds = np.insert(seq_inds, len(seq_inds), EOS)

    struct_list = [*struct]
    struct_inds = np.array([STRUCT_DICT[char] for char in struct_list])
    struct_inds = np.insert(struct_inds, 0, EOS)
    struct_inds = np.insert(struct_inds, len(struct_inds), EOS)

    # seq-struct-inds:
    tmp = ''.join(a + b for a, b in zip(seq, struct))
    seq_struct_inds = np.array([NUCLEOTIDES_STRUCT_DICT[tmp[i:i + 2]] for i in range(0, len(tmp), 2)])
    seq_struct_inds = np.insert(seq_struct_inds, 0, EOS)
    seq_struct_inds = np.insert(seq_struct_inds, len(seq_struct_inds), EOS)
    return pd.Series([seq, s_len, struct, bpp, seq_inds, struct_inds, seq_struct_inds,
                      row['nump_react_a'], row['nump_react_d'], row['error_a'], row['error_d'],
                      row['non_nan_count_a'], row['non_nan_count_d']],
                     index=['seq', 's_len', 'struct', 'bpp', 'seq_inds', 'struct_inds', 'seq_struct_inds',
                            'nump_react_a', 'nump_react_d', 'error_a', 'error_d', 'non_nan_count_a', 'non_nan_count_d'])


def all_info_from_seq_no_bpp(row):
    # received 'sequence', 'nump_react_a', 'nump_react_d', 'error_a', 'error_d', 'non_nan_count_a', 'non_nan_count_d'
    seq = row['sequence'].strip()
    s_len = len(seq)
    struct = mfe(seq, package="eternafold")

    seq_list = [*seq]
    seq_inds = np.array([NUCLEOTIDES_DICT[char] for char in seq_list])
    seq_inds = np.insert(seq_inds, 0, EOS)
    seq_inds = np.insert(seq_inds, len(seq_inds), EOS)

    struct_list = [*struct]
    struct_inds = np.array([STRUCT_DICT[char] for char in struct_list])
    struct_inds = np.insert(struct_inds, 0, EOS)
    struct_inds = np.insert(struct_inds, len(struct_inds), EOS)

    # seq-struct-inds:
    tmp = ''.join(a + b for a, b in zip(seq, struct))
    seq_struct_inds = np.array([NUCLEOTIDES_STRUCT_DICT[tmp[i:i + 2]] for i in range(0, len(tmp), 2)])
    seq_struct_inds = np.insert(seq_struct_inds, 0, EOS)
    seq_struct_inds = np.insert(seq_struct_inds, len(seq_struct_inds), EOS)
    return pd.Series([seq, s_len, struct, seq_inds, struct_inds, seq_struct_inds,
                      row['nump_react_a'], row['nump_react_d'], row['error_a'], row['error_d'],
                      row['non_nan_count_a'], row['non_nan_count_d']],
                     index=['seq', 's_len', 'struct', 'seq_inds', 'struct_inds', 'seq_struct_inds',
                            'nump_react_a', 'nump_react_d', 'error_a', 'error_d', 'non_nan_count_a', 'non_nan_count_d'])


########################################################################
# processing of test sequences for inference:
def process_test_sequences(row):
    seq = row['sequence'].strip()
    s_len = len(seq)
    struct = mfe(seq, package="eternafold")
    # if bpps are used, they would have to be calculated in dataset and not here
    # because test_sequences dataframe is too large (if bpp information is preserved, it would take too much memory)

    seq_list = [*seq]
    seq_inds = np.array([NUCLEOTIDES_DICT[char] for char in seq_list])
    seq_inds = np.insert(seq_inds, 0, EOS)
    seq_inds = np.insert(seq_inds, len(seq_inds), EOS)

    struct_list = [*struct]
    struct_inds = np.array([STRUCT_DICT[char] for char in struct_list])
    struct_inds = np.insert(struct_inds, 0, EOS)
    struct_inds = np.insert(struct_inds, len(struct_inds), EOS)

    # possible to add seq_struct_inds (analogous to all_info_from_seq functions)
    # but seq_struct_inds are not used in this repository and are omitted here to make the resulting dataframe smaller

    id_begin = row['id_min']
    id_end = row['id_max']
    return pd.Series(
        [seq, seq_inds, struct_inds, s_len, id_begin, id_end],
        ['seq', 'seq_inds', 'struct_inds', 's_len', 'id_begin', 'id_end']
    )


