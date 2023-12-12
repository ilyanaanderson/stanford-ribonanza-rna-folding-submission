import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
import json

from models import *
from datasets import *
from seed_all import seed_everything

SUBMISSION_NUMBER = 27  # supports only 27 and 23
MODEL_EPOCH_NUMBER = 1  # 27 for submission number 27, and 44 for submission number 23
# (how many epochs the model was trained, starting from zero)

BATCH = 3

# the second test to see how the model generalizes, according to
# https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653
# only cpu is used for this script (so it's possible to run it at the same time as scripts that require gpu)
if __name__ == '__main__':
    seed_everything()

    # data will be read using the path from json file
    with open('SETTINGS.json') as f:
        data = json.load(f)
    file_to_read = data["GENERALIZATION_DATA"]
    model_dir = data["MODEL_DIR"]
    generalization_dir = data["GENERALIZATION_PICTURES_TWO_DIR"]

    model_string = f"{SUBMISSION_NUMBER}/models/model_{MODEL_EPOCH_NUMBER}.pth"
    model_to_load = os.path.join(model_dir, model_string)

    if SUBMISSION_NUMBER == 27:
        dataset_skeleton = DatasetEightInferGeneralization
        model_skeleton = ModelThirtyNine
        model = model_skeleton()
        num_work = 0
    elif SUBMISSION_NUMBER == 23:
        dataset_skeleton = DatasetTenInferGeneralization
        model_skeleton = ModelThirtyTwo
        model = model_skeleton(num_tokens=LEN_FOR_GENERALIZATION)
        num_work = 40

    df = pd.read_parquet(file_to_read)

    dataset = dataset_skeleton(df=df)
    loader = DataLoader(dataset=dataset, batch_size=BATCH, pin_memory=False, shuffle=False, num_workers=num_work)

    state = torch.load(model_to_load, map_location='cpu')
    model.load_state_dict(state)
    model.eval()

    output_main = torch.empty((0, 722, 2))

    with torch.no_grad():
        i = 0
        for data, ids in loader:
            i += 1
            output = model(data)  # (batch, 722, 2)
            output_main = torch.cat((output_main, output), 0)

            if i % 10 == 0:
                print(f"step {i}")

    m2_preds = output_main[:, 1:-1, :]
    print(m2_preds.shape)

    fig = plt.figure(dpi=500.0)

    # code from https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653
    plt.subplot(121)
    plt.title(f'2A3_for_{SUBMISSION_NUMBER}')
    plt.imshow(m2_preds[:, :, 0], vmin=0, vmax=1, cmap='gray_r')
    plt.subplot(122)
    plt.title(f'DMS_for_{SUBMISSION_NUMBER}')
    plt.imshow(m2_preds[:, :, 1], vmin=0, vmax=1, cmap='gray_r')

    plt.tight_layout()
    file_string = f"{SUBMISSION_NUMBER}_test_two.png"
    path = os.path.join(generalization_dir, file_string)
    plt.savefig(path, dpi=500)
    plt.clf()
    plt.close()

