import os, hashlib, json, numpy as np, random
def set_seeds(seed: int = 42):
    np.random.seed(seed); random.seed(seed)
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except Exception: pass

def hash_features(columns):
    m = hashlib.sha1()
    m.update("|".join(sorted(list(columns))).encode())
    return m.hexdigest()[:12]

def checksum_file(path):
    m = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            m.update(chunk)
    return m.hexdigest()[:12]