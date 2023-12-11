import json
import pandas as pd
from parallel_pandas import ParallelPandas

from data_utils import process_test_sequences

# the script runs for a long time, but eventually finishes
# to speed up, it is possible to re-write this script analogously to how training and validation data are processed
# with an additional step to create partial files (or smaller dataframes) first
if __name__ == '__main__':
    ParallelPandas.initialize(n_cpu=None, split_factor=1, disable_pr_bar=False)
    with open('SETTINGS.json') as f:
        data = json.load(f)
    path_to_read = data["RAW_DATA_TEST"]
    path_to_write = data["TEST_DATA"]

    df = pd.read_csv(path_to_read)  # reaching in chunks is more memory-efficient, but longer
    # if this file is read in chunks, they need to be concatenated afterward

    df = df.p_apply(process_test_sequences, axis=1)
    df.to_parquet(path_to_write, engine='pyarrow', index=False)
