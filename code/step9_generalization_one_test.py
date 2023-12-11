import polars as pl
import matplotlib.pyplot as plt
import os
import json

SUBMISSION_NUMBER = 27

# will generate a figure according to
# https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653
if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        data = json.load(f)
    submission_dir = data["SUBMISSION_DIR"]
    generalization_dir = data["GENERALIZATION_PICTURES_ONE_DIR"]
    path_to_csv_string = f"{SUBMISSION_NUMBER}/{SUBMISSION_NUMBER}.csv"
    path_to_csv = os.path.join(submission_dir, path_to_csv_string)

    # code from https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/444653
    #read your sub here
    df=pl.read_csv(path_to_csv)
    #some parameters
    font_size=6
    id1=269545321
    id2=269724007
    reshape1=391
    reshape2=457
    #get predictions
    pred_DMS=df[id1:id2+1]['reactivity_DMS_MaP'].to_numpy().reshape(reshape1,reshape2)
    pred_2A3=df[id1:id2+1]['reactivity_2A3_MaP'].to_numpy().reshape(reshape1,reshape2)
    #plot mutate and map
    fig = plt.figure()
    plt.subplot(121)
    plt.title(f'reactivity_DMS_MaP, path_number {SUBMISSION_NUMBER}', fontsize=font_size)
    plt.imshow(pred_DMS,vmin=0,vmax=1, cmap='gray_r')
    plt.subplot(122)
    plt.title(f'reactivity_2A3_MaP, path_number {SUBMISSION_NUMBER}', fontsize=font_size)
    plt.imshow(pred_2A3,vmin=0,vmax=1, cmap='gray_r')
    plt.tight_layout()
    file_string = f"{SUBMISSION_NUMBER}.png"
    path = os.path.join(generalization_dir, file_string)
    plt.savefig(path, dpi=500)
    plt.clf()
    plt.close()
