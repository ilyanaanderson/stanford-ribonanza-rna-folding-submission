import os
import json
import gc
from fastai.learner import Learner
from torch.utils.data import DataLoader
from fastai.data.core import DataLoaders
from fastai.callback.schedule import fit_one_cycle
from fastai.callback.training import GradientClip
from fastai.callback.tracker import SaveModelCallback

from losses import *
from models import *
from datasets import *
from seed_all import seed_everything
from data_utils import bpp_to_nump

BATCH = 32
EPOCHS = 45

BPP_YES = False

SUBMISSION_NUMBER = 27   # the setup is shown in this repository for 27 and 23 only
# 27 doesn't need bpps; 23 needs bpps (change BPP_YES variable accordingly)

# before running, the directory ../models/{SUBMISSION_NUMBER}/models needs to already exist
if __name__ == '__main__':
    seed_everything()
    with open('SETTINGS.json') as f:
        data = json.load(f)
    train_data_path = data["TRAIN_DATA"]
    val_data_path = data["VAL_DATA"]
    model_dir = data["MODEL_DIR"]

    if SUBMISSION_NUMBER == 27:
        selected_cols = ['seq_inds', 'struct_inds', 's_len', 'nump_react_a', 'nump_react_d']
        dataset_train_skeleton = DatasetEight
        # choose DatasetTwelve for random perturbations (selected columns will have to include 'error_a' and 'error_d')

        dataset_val_skeleton = DatasetEight
        model_skeleton = ModelThirtyNine

    elif SUBMISSION_NUMBER == 23:
        selected_cols = ['seq_inds', 'struct_inds', 'bpp', 's_len', 'nump_react_a', 'nump_react_d']
        dataset_train_skeleton = DatasetTen
        # choose DatasetEleven for random perturbations (selected columns will have to include 'error_a' and 'error_d')

        dataset_val_skeleton = DatasetTen
        model_skeleton = ModelThirtyTwo

    parent_dir_string = f"{SUBMISSION_NUMBER}"
    parent_dir_for_models_for_learner = os.path.join(model_dir, parent_dir_string)  # this directory needs to exist

    df = pd.read_parquet(train_data_path, engine='pyarrow')

    # drop columns here
    train_df = df[selected_cols]
    del df
    gc.collect()
    print("columns dropped for train df")
    if BPP_YES:
        train_df['bpp'] = train_df['bpp'].apply(bpp_to_nump)
        gc.collect()
        print("bpp pre-processed for train data")

    val_df = pd.read_parquet(val_data_path, engine='pyarrow')
    val_df = val_df[selected_cols]
    print("columns dropped in validation df")
    if BPP_YES:
        val_df['bpp'] = val_df['bpp'].apply(bpp_to_nump)
        gc.collect()
        print("bpp pre-processed for validation data")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    model = model_skeleton()
    model = model.to(device)
    dataset_train = dataset_train_skeleton(df=train_df)
    dataset_val = dataset_val_skeleton(df=val_df)
    loader_train = DataLoader(dataset=dataset_train, batch_size=BATCH, pin_memory=False, shuffle=True)
    loader_val = DataLoader(dataset=dataset_val, batch_size=BATCH, pin_memory=False, shuffle=False)
    dls = DataLoaders(loader_train, loader_val)
    lossObject = LossEight_L1Smooth()
    save_cb = SaveModelCallback(every_epoch=True)  # will save state dicts of models
    learn = Learner(dls=dls, model=model, loss_func=lossObject, path=parent_dir_for_models_for_learner,
                    cbs=[GradientClip(3.0), save_cb]).to_fp16()
    learn.fit_one_cycle(EPOCHS, lr_max=5e-4, wd=0.05, pct_start=0.02)

