import pytest
import numpy as np
import torch

@pytest.fixture(autouse=True, scope='session')
def set_random_seeds():
    seed = 42  # or any other seed value you prefer
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
