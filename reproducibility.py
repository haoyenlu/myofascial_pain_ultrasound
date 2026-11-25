"""Seed a fold before constructing its model and data loaders."""
import os
import random

import numpy as np
import torch


def seed_training(seed):
    seed = int(seed)
    if not 0 <= seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn for unsupported deterministic operations rather than silently
    # promising bitwise reproducibility on every device/PyTorch version.
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def fold_seed(seed, fold):
    return (int(seed) + int(fold) - 1) % (2**32)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
