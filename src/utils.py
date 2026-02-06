from __future__ import annotations

import numpy as np


def ensure_np(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def assert_no_nans(arr: np.ndarray, name: str) -> None:
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaNs.")


def summary(points: np.ndarray, labels: np.ndarray, instance_ids: np.ndarray) -> None:
    n = points.shape[0]
    uniq_labels = np.unique(labels).size
    uniq_inst = np.unique(instance_ids).size
    print(f"Points: {n:,}")
    print(f"Unique semantic labels: {uniq_labels}")
    print(f"Unique instances: {uniq_inst}")


def consistent_random_color_from_id(idx: int) -> np.ndarray:
    seed = (idx * 2654435761) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    color = rng.random(3)
    return color.astype(np.float32)
