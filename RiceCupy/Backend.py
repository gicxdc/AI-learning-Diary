import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT_DIR / ".cupy_cache"))

import cupy as xp
import numpy as cpu_np


def to_cpu(array):
    if isinstance(array, xp.ndarray):
        return xp.asnumpy(array)
    return array


def to_gpu(array, dtype=None):
    return xp.asarray(array, dtype=dtype)
