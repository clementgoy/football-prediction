import numpy as np, random

def set_seeds(seed: int = 42):
    np.random.seed(seed); random.seed(seed)
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except Exception: pass
