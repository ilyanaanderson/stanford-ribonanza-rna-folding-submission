from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import math
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

# datasets for training/validation receive data in format:
# 'seq' is string sequence
# 's_len' is int64 sequence length
# 'struct' is dot-parentheses string
# 'bpp' is 2-D numpy array, of data type float32 (some additional processing is required after parquet file is read)
# note: 'bpp' column can be present or absent
# 'seq_inds' is 3-6 (vocab 7), with eos at the beginning and end
# 'struct_inds' is 3-5 (vocab 6), with eos at the beginning and end
# 'seq_struct_inds' is 3-14 (vocab 15), with eos at the beginning and end
# 'nump_react_a'
# 'nump_react_d'
# 'error_a'
# 'error_d'
# 'non_nan_count_a'
# 'non_nan_count_d'

# eos tokens were added to all inds
# padding is absent
# reactivities not clipped


# this dataset feeds seq_inds as info1, struct_inds as info2, and mask as mask (data dict)
class DatasetEight(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        label = {}

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_struct = row['struct_inds'].copy()
        nump_a = row['nump_react_a'][0:s_len]
        nump_d = row['nump_react_d'][0:s_len]

        # clip reactivities:
        nump_a = np.clip(nump_a, 0, 1)
        nump_d = np.clip(nump_d, 0, 1)

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        struct = torch.from_numpy(nump_struct)
        struct_tensor = F.pad(struct, (pad_left_num, pad_right_num))  # pads with zeros by default

        # work on react_a, react_attn_a
        reactivity_tensor_a = torch.from_numpy(nump_a)  # not padded
        pad_left_num_react = pad_left_num + 1  # because eos was added on the left to sequence
        pad_right_num_for_reactivity = pad_right_num + 1  # 1 added to old number
        react_padded_a = F.pad(reactivity_tensor_a, (pad_left_num_react, pad_right_num_for_reactivity), value=float('nan'))  # padded with nans
        react_attn_a = ~torch.isnan(react_padded_a)
        react_tensor_a = torch.nan_to_num(react_padded_a)  # nans turn into zero (by default)
        # work on react_d, react_attn_d
        reactivity_tensor_d = torch.from_numpy(nump_d)  # not padded
        react_padded_d = F.pad(reactivity_tensor_d, (pad_left_num_react, pad_right_num_for_reactivity),
                               value=float('nan'))  # padded with nans
        react_attn_d = ~torch.isnan(react_padded_d)
        react_tensor_d = torch.nan_to_num(react_padded_d)  # nans turn into zero (by default)

        data['info1'] = seq_tensor
        data['info2'] = struct_tensor
        data['mask'] = mask
        label['react_a'] = react_tensor_a
        label['react_attn_a'] = react_attn_a
        label['react_d'] = react_tensor_d
        label['react_attn_d'] = react_attn_d
        return data, label


# if it was trained with DatasetEight or DatasetTwelve, DatasetEightInfer is used for inference
class DatasetEightInfer(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        ids = torch.zeros(4, dtype=torch.int64)

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_struct = row['struct_inds'].copy()

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        struct = torch.from_numpy(nump_struct)
        struct_tensor = F.pad(struct, (pad_left_num, pad_right_num))  # pads with zeros by default

        # work on ids
        ids[0] = row['id_begin']
        ids[1] = row['id_end']
        ids[2] = pad_left_num + 1  # slice start (adding 1 because 'eos' token was added, but we don't collect reactivity from it)
        ids[3] = s_len

        data['info1'] = seq_tensor
        data['info2'] = struct_tensor
        data['mask'] = mask

        return data, ids


# for testing generalization of models trained with DatasetEight or DatasetTwelve
class DatasetEightInferGeneralization(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        ids = 0.0

        seq = row['sequence']
        struct = row['structure']

        seq_list = [*seq]
        seq_inds = np.array([NUCLEOTIDES_DICT[char] for char in seq_list])
        seq_inds = np.insert(seq_inds, 0, EOS)
        nump_seq = np.insert(seq_inds, len(seq_inds), EOS)

        struct_list = [*struct]
        struct_inds = np.array([STRUCT_DICT[char] for char in struct_list])
        struct_inds = np.insert(struct_inds, 0, EOS)
        nump_struct = np.insert(struct_inds, len(struct_inds), EOS)

        seq_tensor = torch.from_numpy(nump_seq)
        mask = seq_tensor != 0

        struct_tensor = torch.from_numpy(nump_struct)

        data['info1'] = seq_tensor
        data['info2'] = struct_tensor
        data['mask'] = mask

        return data, ids


# this dataset feeds seq_inds as info1, bpp as info2, and mask as mask (data dict). Label stays the same
# bpp coming into dataset is one numpy array (2-D array) of correct dtype (float32)
class DatasetNine(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        label = {}

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_bpp = row['bpp']
        nump_a = row['nump_react_a'][0:s_len]
        nump_d = row['nump_react_d'][0:s_len]

        # clip reactivities:
        nump_a = np.clip(nump_a, 0, 1)
        nump_d = np.clip(nump_d, 0, 1)

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        bpp = torch.from_numpy(nump_bpp)
        bpp_tensor = F.pad(bpp, (pad_left_num+1, pad_right_num+1, pad_left_num+1, pad_right_num+1))  # pads with zeros by default

        # work on react_a, react_attn_a
        reactivity_tensor_a = torch.from_numpy(nump_a)  # not padded
        pad_left_num_react = pad_left_num + 1  # because eos was added on the left to sequence
        pad_right_num_for_reactivity = pad_right_num + 1  # 1 added to old number
        react_padded_a = F.pad(reactivity_tensor_a, (pad_left_num_react, pad_right_num_for_reactivity), value=float('nan'))  # padded with nans
        react_attn_a = ~torch.isnan(react_padded_a)
        react_tensor_a = torch.nan_to_num(react_padded_a)  # nans turn into zero (by default)
        # work on react_d, react_attn_d
        reactivity_tensor_d = torch.from_numpy(nump_d)  # not padded
        react_padded_d = F.pad(reactivity_tensor_d, (pad_left_num_react, pad_right_num_for_reactivity),
                               value=float('nan'))  # padded with nans
        react_attn_d = ~torch.isnan(react_padded_d)
        react_tensor_d = torch.nan_to_num(react_padded_d)  # nans turn into zero (by default)

        data['info1'] = seq_tensor
        data['info2'] = bpp_tensor
        data['mask'] = mask
        label['react_a'] = react_tensor_a
        label['react_attn_a'] = react_attn_a
        label['react_d'] = react_tensor_d
        label['react_attn_d'] = react_attn_d
        return data, label


# this dataset feeds seq_inds as info1, bpp as info2, struct_inds as info3, and mask as mask (data dict). Label the same
# it is a combination of DatasetEight and DatasetNine
class DatasetTen(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        label = {}

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_struct = row['struct_inds'].copy()
        nump_bpp = row['bpp']
        nump_a = row['nump_react_a'][0:s_len]
        nump_d = row['nump_react_d'][0:s_len]

        # clip reactivities:
        nump_a = np.clip(nump_a, 0, 1)
        nump_d = np.clip(nump_d, 0, 1)

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        struct = torch.from_numpy(nump_struct)
        struct_tensor = F.pad(struct, (pad_left_num, pad_right_num))  # pads with zeros by default

        bpp = torch.from_numpy(nump_bpp)
        bpp_tensor = F.pad(bpp, (pad_left_num + 1, pad_right_num + 1, pad_left_num + 1, pad_right_num + 1))

        # work on react_a, react_attn_a
        reactivity_tensor_a = torch.from_numpy(nump_a)  # not padded
        pad_left_num_react = pad_left_num + 1  # because eos was added on the left to sequence
        pad_right_num_for_reactivity = pad_right_num + 1  # 1 added to old number
        react_padded_a = F.pad(reactivity_tensor_a, (pad_left_num_react, pad_right_num_for_reactivity), value=float('nan'))  # padded with nans
        react_attn_a = ~torch.isnan(react_padded_a)
        react_tensor_a = torch.nan_to_num(react_padded_a)  # nans turn into zero (by default)
        # work on react_d, react_attn_d
        reactivity_tensor_d = torch.from_numpy(nump_d)  # not padded
        react_padded_d = F.pad(reactivity_tensor_d, (pad_left_num_react, pad_right_num_for_reactivity),
                               value=float('nan'))  # padded with nans
        react_attn_d = ~torch.isnan(react_padded_d)
        react_tensor_d = torch.nan_to_num(react_padded_d)  # nans turn into zero (by default)

        data['info1'] = seq_tensor
        data['info2'] = bpp_tensor
        data['info3'] = struct_tensor
        data['mask'] = mask
        label['react_a'] = react_tensor_a
        label['react_attn_a'] = react_attn_a
        label['react_d'] = react_tensor_d
        label['react_attn_d'] = react_attn_d
        return data, label


# for inference for models trained with DatasetTen or DatasetEleven
class DatasetTenInfer(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        ids = torch.zeros(4, dtype=torch.int64)

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_struct = row['struct_inds'].copy()

        # calculating nump_bpp
        sequence = row['seq']
        nump_bpp = bpps(sequence, package="eternafold")
        nump_bpp = np.float32(nump_bpp)

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        struct = torch.from_numpy(nump_struct)
        struct_tensor = F.pad(struct, (pad_left_num, pad_right_num))  # pads with zeros by default

        bpp = torch.from_numpy(nump_bpp)
        bpp_tensor = F.pad(bpp, (pad_left_num + 1, pad_right_num + 1, pad_left_num + 1, pad_right_num + 1))

        # work on ids
        ids[0] = row['id_begin']
        ids[1] = row['id_end']
        ids[2] = pad_left_num + 1  # slice start (adding 1 because 'eos' token was added, but we don't collect reactivity from it)
        ids[3] = s_len

        data['info1'] = seq_tensor
        data['info2'] = bpp_tensor
        data['info3'] = struct_tensor
        data['mask'] = mask

        return data, ids


# for generalization test for models trained with DatasetTen or DatasetEleven
class DatasetTenInferGeneralization(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        ids = 0.0

        seq = row['sequence']
        struct = row['structure']

        seq_list = [*seq]
        seq_inds = np.array([NUCLEOTIDES_DICT[char] for char in seq_list])
        seq_inds = np.insert(seq_inds, 0, EOS)
        nump_seq = np.insert(seq_inds, len(seq_inds), EOS)

        struct_list = [*struct]
        struct_inds = np.array([STRUCT_DICT[char] for char in struct_list])
        struct_inds = np.insert(struct_inds, 0, EOS)
        nump_struct = np.insert(struct_inds, len(struct_inds), EOS)

        nump_bpp = bpps(seq, package="eternafold")
        nump_bpp = np.float32(nump_bpp)

        seq_tensor = torch.from_numpy(nump_seq)
        mask = seq_tensor != 0

        bpp = torch.from_numpy(nump_bpp)
        bpp_tensor = F.pad(bpp, (1, 1, 1, 1))

        struct_tensor = torch.from_numpy(nump_struct)

        data['info1'] = seq_tensor
        data['info2'] = bpp_tensor
        data['info3'] = struct_tensor
        data['mask'] = mask

        return data, ids


# similar to DatasetTen, but can perturb targets according to errors
class DatasetEleven(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        label = {}

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_struct = row['struct_inds'].copy()
        nump_bpp = row['bpp']
        nump_a = row['nump_react_a'][0:s_len]
        nump_d = row['nump_react_d'][0:s_len]

        error_a = row['error_a'][0:s_len]
        error_d = row['error_d'][0:s_len]
        error_a = (np.nan_to_num(error_a, copy=True, nan=0.0)) / 2.0
        error_d = (np.nan_to_num(error_d, copy=True, nan=0.0)) / 2.0
        multiplier_a = np.float32(np.random.rand(s_len) * 2 - 1)
        multiplier_d = np.float32(np.random.rand(s_len) * 2 - 1)
        perturb_a = error_a * multiplier_a
        perturb_d = error_d * multiplier_d

        nump_a = nump_a + perturb_a
        nump_d = nump_d + perturb_d

        # clip reactivities:
        nump_a = np.clip(nump_a, 0, 1)
        nump_d = np.clip(nump_d, 0, 1)

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        struct = torch.from_numpy(nump_struct)
        struct_tensor = F.pad(struct, (pad_left_num, pad_right_num))  # pads with zeros by default

        bpp = torch.from_numpy(nump_bpp)
        bpp_tensor = F.pad(bpp, (pad_left_num + 1, pad_right_num + 1, pad_left_num + 1, pad_right_num + 1))

        # work on react_a, react_attn_a
        reactivity_tensor_a = torch.from_numpy(nump_a)  # not padded
        pad_left_num_react = pad_left_num + 1  # because eos was added on the left to sequence
        pad_right_num_for_reactivity = pad_right_num + 1  # 1 added to old number
        react_padded_a = F.pad(reactivity_tensor_a, (pad_left_num_react, pad_right_num_for_reactivity), value=float('nan'))  # padded with nans
        react_attn_a = ~torch.isnan(react_padded_a)
        react_tensor_a = torch.nan_to_num(react_padded_a)  # nans turn into zero (by default)
        # work on react_d, react_attn_d
        reactivity_tensor_d = torch.from_numpy(nump_d)  # not padded
        react_padded_d = F.pad(reactivity_tensor_d, (pad_left_num_react, pad_right_num_for_reactivity),
                               value=float('nan'))  # padded with nans
        react_attn_d = ~torch.isnan(react_padded_d)
        react_tensor_d = torch.nan_to_num(react_padded_d)  # nans turn into zero (by default)

        data['info1'] = seq_tensor
        data['info2'] = bpp_tensor
        data['info3'] = struct_tensor
        data['mask'] = mask
        label['react_a'] = react_tensor_a
        label['react_attn_a'] = react_attn_a
        label['react_d'] = react_tensor_d
        label['react_attn_d'] = react_attn_d
        return data, label


# this dataset feeds seq_inds as info1, struct_inds as info2, and mask as mask (data dict). Label stays the same
# same as DatasetEight, but it can perturb reactivities in the same way as DatasetEleven does
class DatasetTwelve(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = {}
        label = {}

        s_len = row['s_len']
        nump_seq = row['seq_inds'].copy()
        nump_struct = row['struct_inds'].copy()
        nump_a = row['nump_react_a'][0:s_len]
        nump_d = row['nump_react_d'][0:s_len]

        error_a = row['error_a'][0:s_len]
        error_d = row['error_d'][0:s_len]
        error_a = (np.nan_to_num(error_a, copy=True, nan=0.0)) / 2.0
        error_d = (np.nan_to_num(error_d, copy=True, nan=0.0)) / 2.0
        multiplier_a = np.float32(np.random.rand(s_len) * 2 - 1)
        multiplier_d = np.float32(np.random.rand(s_len) * 2 - 1)
        perturb_a = error_a * multiplier_a
        perturb_d = error_d * multiplier_d

        nump_a = nump_a + perturb_a
        nump_d = nump_d + perturb_d

        # clip reactivities:
        nump_a = np.clip(nump_a, 0, 1)
        nump_d = np.clip(nump_d, 0, 1)

        seq = torch.from_numpy(nump_seq)
        pad_left_num = math.trunc((TARGET_LEN - s_len) / 2)  # left as is since 2 was added to target_len and to s_len
        pad_right_num = TARGET_LEN - s_len - pad_left_num  # left as is
        seq_tensor = F.pad(seq, (pad_left_num, pad_right_num))  # pads with zeros by default
        mask = seq_tensor != 0

        struct = torch.from_numpy(nump_struct)
        struct_tensor = F.pad(struct, (pad_left_num, pad_right_num))  # pads with zeros by default

        # work on react_a, react_attn_a
        reactivity_tensor_a = torch.from_numpy(nump_a)  # not padded
        pad_left_num_react = pad_left_num + 1  # because eos was added on the left to sequence
        pad_right_num_for_reactivity = pad_right_num + 1  # 1 added to old number
        react_padded_a = F.pad(reactivity_tensor_a, (pad_left_num_react, pad_right_num_for_reactivity), value=float('nan'))  # padded with nans
        react_attn_a = ~torch.isnan(react_padded_a)
        react_tensor_a = torch.nan_to_num(react_padded_a)  # nans turn into zero (by default)
        # work on react_d, react_attn_d
        reactivity_tensor_d = torch.from_numpy(nump_d)  # not padded
        react_padded_d = F.pad(reactivity_tensor_d, (pad_left_num_react, pad_right_num_for_reactivity),
                               value=float('nan'))  # padded with nans
        react_attn_d = ~torch.isnan(react_padded_d)
        react_tensor_d = torch.nan_to_num(react_padded_d)  # nans turn into zero (by default)

        data['info1'] = seq_tensor
        data['info2'] = struct_tensor
        data['mask'] = mask
        label['react_a'] = react_tensor_a
        label['react_attn_a'] = react_attn_a
        label['react_d'] = react_tensor_d
        label['react_attn_d'] = react_attn_d
        return data, label

