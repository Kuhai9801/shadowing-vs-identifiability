"""Reproducibility helpers shared by entrypoint scripts."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def seed_everything(seed: int, *, deterministic_torch: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs with best-effort determinism."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Use a stable cuBLAS workspace config when deterministic algorithms are requested.
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
