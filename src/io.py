from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import open3d as o3d
from .utils import ensure_np, assert_no_nans, summary
import os


def _find_attribute_key(pc_t: "o3d.t.geometry.PointCloud", candidates: Tuple[str, ...]) -> Optional[str]:
    for k in candidates:
        if k in pc_t.point:
            return k
    return None


def load_pointcloud(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise ValueError(f"Point cloud file is missing or empty: {path}")
    pc_t = o3d.t.io.read_point_cloud(path)
    if "positions" not in pc_t.point:
        raise ValueError("PLY missing 'positions' attribute (Open3D tensor expects positions).")
    points = ensure_np(pc_t.point["positions"]).astype(np.float32)
    color_key = _find_attribute_key(pc_t, ("colors", "color", "rgb"))
    if color_key is None:
        colors = np.full_like(points, 0.7, dtype=np.float32)
    else:
        colors = ensure_np(pc_t.point[color_key]).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    sem_key = _find_attribute_key(pc_t, ("semantic", "label", "labels", "class", "semantic_label", "semantic_labels"))
    if sem_key is None:
        raise ValueError(
            "Could not find semantic label attribute. Tried: semantic/label/labels/class/semantic_label/semantic_labels"
        )
    labels = ensure_np(pc_t.point[sem_key]).reshape(-1).astype(np.int32)
    inst_key = _find_attribute_key(pc_t, ("instance", "instance_id", "instance_ids", "object_id", "obj_id"))
    if inst_key is None:
        raise ValueError(
            "Could not find instance id attribute. Tried: instance/instance_id/instance_ids/object_id/obj_id"
        )
    instance_ids = ensure_np(pc_t.point[inst_key]).reshape(-1).astype(np.int32)

    return {
        "points": points,
        "colors": colors,
        "labels": labels,
        "instance_ids": instance_ids,
    }


def validate_data(pc: Dict[str, np.ndarray]) -> None:
    """Check shapes, NaNs, and print a summary."""
    pts = pc["points"]
    cols = pc["colors"]
    labels = pc["labels"]
    inst = pc["instance_ids"]

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")

    n = pts.shape[0]

    if cols.shape != (n, 3):
        raise ValueError(f"colors must be (N,3), got {cols.shape}")

    if labels.shape != (n,):
        raise ValueError(f"labels must be (N,), got {labels.shape}")

    if inst.shape != (n,):
        raise ValueError(f"instance_ids must be (N,), got {inst.shape}")

    assert_no_nans(pts, "points")
    assert_no_nans(cols, "colors")

    summary(pts, labels, inst)
