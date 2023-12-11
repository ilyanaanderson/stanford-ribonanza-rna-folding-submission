import gc
import os
import time
import pandas as pd
import numpy as np
import json
import torch
from fastai.data.load import DataLoader

from datasets import DatasetEightInfer, DatasetTenInfer
from models import ModelThirtyNine, ModelThirtyTwo
from seed_all import seed_everything

SUBMISSION_NUMBER = 27  # the setup is shown in this repository for 27 and 23 only
MODEL_EPOCH_NUMBER = 27  # 27 for submission number 27, and 44 for submission number 23
# (how many epochs the model was trained, starting from zero)

BATCH = 128
COL_A = 'reactivity_2A3_MaP'
COL_D = 'reactivity_DMS_MaP'


def batch_to_csv(output, ids, main_path_for_parquets):
    # received a batch of outputs (B, 459, 2) and ids (B, 4) as numpy arrays
    name_of_csv = ids[0][0]
    dfs = []
    for i in range(output.shape[0]):
        start_id = ids[i][0]
        end_id = ids[i][1]
        start_index = ids[i][2]
        num_reactivities = ids[i][3]
        # Extract relevant reactivities from output[i]
        reactivities_a = output[i, start_index: start_index + num_reactivities, 0]
        reactivities_d = output[i, start_index: start_index + num_reactivities, 1]
        # Create a DataFrame for the current datapoint
        datapoint_df = pd.DataFrame({
            'id': np.arange(start_id, end_id + 1),
            COL_D: reactivities_d,
            COL_A: reactivities_a
        })
        dfs.append(datapoint_df)
    small_df = pd.concat(dfs, ignore_index=True)
    # the df will be written into .parquet
    path = os.path.join(main_path_for_parquets, f"{name_of_csv}.parquet")
    small_df.to_parquet(path, index=False, engine='pyarrow')
    return


# before running, folder ../submissions/{SUBMISSION_NUMBER}/all needs to already exist
if __name__ == '__main__':
    seed_everything()
    with open('SETTINGS.json') as f:
        data = json.load(f)
    path_to_test_data = data["TEST_DATA"]
    model_dir = data["MODEL_DIR"]
    submission_dir = data["SUBMISSION_DIR"]
    model_string = f"{SUBMISSION_NUMBER}/models/model_{MODEL_EPOCH_NUMBER}.pth"
    path_to_model = os.path.join(model_dir, model_string)
    main_path_string = f"{SUBMISSION_NUMBER}/all/"
    main_path_for_parquets = os.path.join(submission_dir, main_path_string)

    df = pd.read_parquet(path_to_test_data, engine='pyarrow')
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    if SUBMISSION_NUMBER == 27:
        dataset_skeleton = DatasetEightInfer
        model_skeleton = ModelThirtyNine
    elif SUBMISSION_NUMBER == 23:
        dataset_skeleton = DatasetTenInfer
        model_skeleton = ModelThirtyTwo

    # dataset and dataloader
    dataset = dataset_skeleton(df=df)
    loader = DataLoader(dataset=dataset, batch_size=BATCH, pin_memory=False, shuffle=False, device=device,
                        num_workers=40)  # num_workers is set to 40 for bpps (submission number 23)

    # model
    model = model_skeleton()
    # load the state dict
    model.load_state_dict(torch.load(path_to_model))
    model.eval()
    model.to(device)

    # Start timer
    start_time = time.time()
    with torch.no_grad():
        i = 0
        for data, ids in loader:
            i += 1
            out = model(data)
            batch_to_csv(out.detach().cpu().numpy(), ids.detach().cpu().numpy(), main_path_for_parquets)
            if i % 50 == 0:
                print(f"step {i}")
    # End timer
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)


