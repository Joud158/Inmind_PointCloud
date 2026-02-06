from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import open3d as o3d
from .utils import consistent_random_color_from_id
from .processing import voxel_downsample, crop_axis_aligned, center


PointCloudDict = Dict[str, np.ndarray]


def color_by_semantic_label(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32).reshape(-1)
    unique = np.unique(labels)
    color_map = {}
    for lab in unique:
        color_map[int(lab)] = consistent_random_color_from_id(int(lab))
    colors = np.vstack([color_map[int(l)] for l in labels]).astype(np.float32)
    return colors


def color_by_instance_id(instance_ids: np.ndarray) -> np.ndarray:
    ids = instance_ids.astype(np.int32).reshape(-1)
    unique = np.unique(ids)
    color_map = {}
    for iid in unique:
        color_map[int(iid)] = consistent_random_color_from_id(int(iid) + 10_000)  # offset from semantic ids
    colors = np.vstack([color_map[int(i)] for i in ids]).astype(np.float32)
    return colors


def show_only_class(pc: PointCloudDict, labels: np.ndarray, class_id: int) -> PointCloudDict:
    cid = int(class_id)
    mask = (labels.reshape(-1) == cid)
    return {
        "points": pc["points"][mask],
        "colors": pc["colors"][mask],
        "labels": pc["labels"][mask],
        "instance_ids": pc["instance_ids"][mask],
    }


def to_o3d_geometry(pc: PointCloudDict, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc["points"].astype(np.float64))
    use_cols = pc["colors"] if colors is None else colors
    pcd.colors = o3d.utility.Vector3dVector(use_cols.astype(np.float64))
    return pcd

@dataclass
class ViewerState:
    original: PointCloudDict
    current: PointCloudDict
    mode: str = "rgb"
    downsample_on: bool = False
    voxel_size: float = 0.05
    crop_on: bool = False
    crop_scale: float = 1.0
    point_size: float = 2.0


def _apply_pipeline(state: ViewerState) -> PointCloudDict:
    pc = state.current
    if state.crop_on:
        pts = pc["points"]
        minb = pts.min(axis=0)
        maxb = pts.max(axis=0)
        center_pt = 0.5 * (minb + maxb)
        half = 0.5 * (maxb - minb) * float(state.crop_scale)
        minc = center_pt - half
        maxc = center_pt + half
        pc = crop_axis_aligned(pc, minc, maxc)
    if state.downsample_on:
        pc = voxel_downsample(pc, state.voxel_size)
    return pc


def _colors_for_mode(pc: PointCloudDict, mode: str) -> np.ndarray:
    if mode == "rgb":
        return pc["colors"]
    if mode == "semantic":
        return color_by_semantic_label(pc["labels"])
    if mode == "instance":
        return color_by_instance_id(pc["instance_ids"])
    return pc["colors"]


def interactive_viewer(pc: PointCloudDict) -> None:
    state = ViewerState(original={k: v.copy() for k, v in pc.items()},
                        current={k: v.copy() for k, v in pc.items()})
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Open3D Interactive Viewer", width=1280, height=720)

    render_opt = vis.get_render_option()
    render_opt.point_size = float(state.point_size)
    piped = _apply_pipeline(state)
    cols = _colors_for_mode(piped, state.mode)
    geom = to_o3d_geometry(piped, cols)
    vis.add_geometry(geom)

    def _refresh():
        nonlocal geom
        vis.remove_geometry(geom, reset_bounding_box=False)
        piped2 = _apply_pipeline(state)
        cols2 = _colors_for_mode(piped2, state.mode)
        geom = to_o3d_geometry(piped2, cols2)
        vis.add_geometry(geom, reset_bounding_box=False)
        render_opt.point_size = float(state.point_size)
        vis.update_renderer()

    def set_mode_rgb(v):
        state.mode = "rgb"
        _refresh()
        return False

    def set_mode_sem(v):
        state.mode = "semantic"
        _refresh()
        return False

    def set_mode_inst(v):
        state.mode = "instance"
        _refresh()
        return False

    def toggle_downsample(v):
        state.downsample_on = not state.downsample_on
        _refresh()
        return False

    def voxel_plus(v):
        state.voxel_size = float(state.voxel_size * 1.25)
        _refresh()
        return False

    def voxel_minus(v): 
        state.voxel_size = float(max(1e-4, state.voxel_size / 1.25))
        _refresh()
        return False

    def do_center(v): 
        state.current = center(state.current)
        _refresh()
        return False

    def toggle_crop(v): 
        state.crop_on = not state.crop_on
        _refresh()
        return False

    def crop_shrink(v): 
        state.crop_scale = float(max(0.05, state.crop_scale * 0.85))
        _refresh()
        return False

    def crop_expand(v): 
        state.crop_scale = float(min(1.0, state.crop_scale / 0.85))
        _refresh()
        return False

    def reset_all(v):  
        state.current = {k: v.copy() for k, v in state.original.items()}
        state.mode = "rgb"
        state.downsample_on = False
        state.voxel_size = 0.05
        state.crop_on = False
        state.crop_scale = 1.0
        _refresh()
        return False

    def reset_view(v):  
        vis.reset_view_point(True)
        return False

    vis.register_key_callback(ord("1"), set_mode_rgb)
    vis.register_key_callback(ord("2"), set_mode_sem)
    vis.register_key_callback(ord("3"), set_mode_inst)

    vis.register_key_callback(ord("D"), toggle_downsample)
    vis.register_key_callback(ord("C"), do_center)
    vis.register_key_callback(ord("R"), reset_all)
    vis.register_key_callback(ord("V"), reset_view)
    vis.register_key_callback(ord("X"), toggle_crop)

    vis.register_key_callback(ord("+"), voxel_plus)
    vis.register_key_callback(ord("="), voxel_plus)   
    vis.register_key_callback(ord("-"), voxel_minus)
    vis.register_key_callback(ord("_"), voxel_minus)

    vis.register_key_callback(ord("["), crop_shrink)
    vis.register_key_callback(ord("]"), crop_expand)

    print("\nControls:")
    print("  1 RGB | 2 Semantic | 3 Instance")
    print("  D toggle downsample | +/- voxel size")
    print("  X toggle crop | [ ] crop size")
    print("  C center | R reset pipeline | V reset camera")
    print("  Q/Esc close window\n")

    vis.run()
    vis.destroy_window()
