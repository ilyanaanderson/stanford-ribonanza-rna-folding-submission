import random
import os
import numpy as np
import torch


# from https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb with modifications
def seed_everything(seed=2837490):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # https://towardsdatascience.com/creating-a-plant-pet-toxicity-classifier-13b8ba6289e6
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) # throws an error in inference
