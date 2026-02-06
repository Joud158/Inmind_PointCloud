from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import open3d as o3d


PointCloudDict = Dict[str, np.ndarray]


def _copy_pc(pc: PointCloudDict) -> PointCloudDict:
    return {k: v.copy() for k, v in pc.items()}

def scale(pc: PointCloudDict, factor: float) -> PointCloudDict:
    out = _copy_pc(pc)
    out["points"] = out["points"] * float(factor)
    return out

def translate(pc: PointCloudDict, offset: np.ndarray) -> PointCloudDict:
    out = _copy_pc(pc)
    off = np.asarray(offset, dtype=np.float32).reshape(1, 3)
    out["points"] = out["points"] + off
    return out

def center(pc: PointCloudDict) -> PointCloudDict:
    out = _copy_pc(pc)
    centroid = out["points"].mean(axis=0, keepdims=True)
    out["points"] = out["points"] - centroid
    return out

def crop_axis_aligned(pc: PointCloudDict, min_bound, max_bound) -> PointCloudDict:
    out = _copy_pc(pc)
    minb = np.asarray(min_bound, dtype=np.float32).reshape(1, 3)
    maxb = np.asarray(max_bound, dtype=np.float32).reshape(1, 3)

    pts = out["points"]
    mask = np.all((pts >= minb) & (pts <= maxb), axis=1)

    out["points"] = out["points"][mask]
    out["colors"] = out["colors"][mask]
    out["labels"] = out["labels"][mask]
    out["instance_ids"] = out["instance_ids"][mask]
    return out

def voxel_downsample(pc: PointCloudDict, voxel_size: float) -> PointCloudDict:
    voxel = float(voxel_size)
    if voxel <= 0:
        return _copy_pc(pc)
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(pc["points"].astype(np.float64))
    src_pcd.colors = o3d.utility.Vector3dVector(pc["colors"].astype(np.float64))

    ds_pcd = src_pcd.voxel_down_sample(voxel_size=voxel)
    ds_pts = np.asarray(ds_pcd.points, dtype=np.float32)
    ds_cols = np.asarray(ds_pcd.colors, dtype=np.float32)

    if ds_pts.shape[0] == 0:
        return {
            "points": ds_pts,
            "colors": ds_cols,
            "labels": np.array([], dtype=np.int32),
            "instance_ids": np.array([], dtype=np.int32),
        }

    kdtree = o3d.geometry.KDTreeFlann(src_pcd)

    labels = pc["labels"].astype(np.int32)
    inst = pc["instance_ids"].astype(np.int32)

    ds_labels = np.empty((ds_pts.shape[0],), dtype=np.int32)
    ds_inst = np.empty((ds_pts.shape[0],), dtype=np.int32)

    for i in range(ds_pts.shape[0]):
        q = ds_pts[i].astype(np.float64) 
        _, idxs, _ = kdtree.search_knn_vector_3d(q, 1)
        j = int(idxs[0])
        ds_labels[i] = labels[j]
        ds_inst[i] = inst[j]

    return {
        "points": ds_pts,
        "colors": ds_cols,
        "labels": ds_labels,
        "instance_ids": ds_inst,
    }

def uniform_downsample(pc: PointCloudDict, every_k_points: int) -> PointCloudDict:
    k = int(every_k_points)
    if k <= 1:
        return _copy_pc(pc)

    out = _copy_pc(pc)
    idx = np.arange(out["points"].shape[0])[::k]

    out["points"] = out["points"][idx]
    out["colors"] = out["colors"][idx]
    out["labels"] = out["labels"][idx]
    out["instance_ids"] = out["instance_ids"][idx]
    return out

def remove_statistical_outliers(
    pc: PointCloudDict,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0
) -> PointCloudDict:

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc["points"])
    pcd.colors = o3d.utility.Vector3dVector(pc["colors"])
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=int(nb_neighbors), std_ratio=float(std_ratio))
    ind = np.asarray(ind, dtype=np.int64)

    out = _copy_pc(pc)
    out["points"] = out["points"][ind]
    out["colors"] = out["colors"][ind]
    out["labels"] = out["labels"][ind]
    out["instance_ids"] = out["instance_ids"][ind]
    return out
