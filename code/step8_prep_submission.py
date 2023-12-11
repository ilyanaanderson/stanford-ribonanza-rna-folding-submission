import dask.dataframe as dd
import pandas as pd
import json
import gc

from seed_all import *

SUBMISSION_NUMBER = 27  # the setup is shown in this repository for 27 and 23 only


# to create the submission
if __name__ == '__main__':
    seed_everything()
    with open('SETTINGS.json') as f:
        data = json.load(f)
    submission_dir = data["SUBMISSION_DIR"]
    path_to_read_string = f"{SUBMISSION_NUMBER}/all/*.parquet"
    path_to_read = os.path.join(submission_dir, path_to_read_string)
    path_to_write_string = f"{SUBMISSION_NUMBER}/{SUBMISSION_NUMBER}.csv"
    path_to_write = os.path.join(submission_dir, path_to_write_string)

    ddf = dd.read_parquet(path_to_read)

    df = ddf.compute()

    df = df.sort_values(by='id', ignore_index=True)
    print('sorted')

    gc.collect()

    # now need to clip the values (just in case)
    df['reactivity_DMS_MaP'] = df['reactivity_DMS_MaP'].clip(0, 1)
    df['reactivity_2A3_MaP'] = df['reactivity_2A3_MaP'].clip(0, 1)

    print(" ")
    print("final df look:")
    print(df.head(7))
    print(df.tail(5))
    print(len(df))

    df.to_csv(path_to_write, index=False)


