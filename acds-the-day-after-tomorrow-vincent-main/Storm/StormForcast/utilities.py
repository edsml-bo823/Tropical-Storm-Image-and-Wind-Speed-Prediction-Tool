import random
import numpy as np
import torch


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out
    any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # uses the inbuilt cudnn auto-tuner to find the fastest convolution
    # algorithms. -
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    return True
