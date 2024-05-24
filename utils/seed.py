import torch
import numpy as np
import random
import dgl

def set_seed(seed):
    ### Torch Seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ### Numpy Seed
    np.random.seed(seed)

    ### Python Seed
    random.seed(seed)

    ### DGL Seed
    dgl.seed(seed)