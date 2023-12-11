import pandas as pd
import json

from data_utils import process_generalization


# it runs a little long, about half an hour (to speed up, possible to use ParallelPandas)
# however, when it runs slow, it doesn't use a lot of resources: possible to run at the same time with training
if __name__ == '__main__':
    with open('SETTINGS.json') as f:
        data = json.load(f)
    file_to_read = data["RAW_GENERALIZATION_DATA"]
    file_to_write = data["GENERALIZATION_DATA"]
    df = pd.read_csv(file_to_read)

    df = df.apply(process_generalization, axis=1)
    df.to_parquet(file_to_write, engine='pyarrow', index=False)
