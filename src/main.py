from __future__ import annotations
import argparse
from .io import load_pointcloud, validate_data
from .processing import center, voxel_downsample, remove_statistical_outliers
from .visualization import interactive_viewer


def build_pipeline(pc_dict, voxel_size: float):
    pc = center(pc_dict)
    pc = voxel_downsample(pc, voxel_size)
    pc = remove_statistical_outliers(pc, nb_neighbors=20, std_ratio=2.0)
    return pc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to .ply point cloud")
    parser.add_argument("--voxel", type=float, default=0.05, help="Initial voxel size for pipeline")
    args = parser.parse_args()

    pc = load_pointcloud(args.path)
    validate_data(pc)

    pc2 = build_pipeline(pc, args.voxel)

    interactive_viewer(pc2)


if __name__ == "__main__":
    main()
