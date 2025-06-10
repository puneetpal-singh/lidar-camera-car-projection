from __future__ import annotations

import argparse
from pathlib import Path

import json
import numpy as np
import open3d as o3d
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate projected car clusters against 3D boxes.")
    parser.add_argument("--results-dir", default=r"D:\M\Projects\KITTI-360_sample\Lidar-Project\output_yolo_gt_ply")
    return parser.parse_args()


def points_in_obb(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    center = np.mean(corners, axis=0)
    x_axis = normalize(corners[1] - corners[0])
    y_axis = normalize(corners[3] - corners[1])
    z_axis = normalize(corners[4] - corners[0])
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    local_points = (points - center) @ rotation
    extents = np.array([
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[3] - corners[1]),
        np.linalg.norm(corners[4] - corners[0]),
    ]) / 2
    return np.all(np.abs(local_points) <= extents, axis=1)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def evaluate_scene(results_dir: Path, scene_id: str) -> pd.DataFrame:
    point_cloud = o3d.io.read_point_cloud(str(results_dir / f"{scene_id}_yolo_gt.ply"))
    points = np.asarray(point_cloud.points)
    assignment = np.load(results_dir / f"{scene_id}_assignment.npy")
    boxes = json.loads((results_dir / f"{scene_id}_gt_boxes.json").read_text())

    rows = []
    for car_id in np.unique(assignment):
        if car_id == -1:
            continue

        mask_points = points[assignment == car_id]
        inside_count = sum(np.sum(points_in_obb(mask_points, np.array(box))) for box in boxes)
        if inside_count == 0:
            continue

        total_points = len(mask_points)
        rows.append({
            "car_id": int(car_id),
            "mask_points": int(total_points),
            "inside_points": int(inside_count),
            "bleed_out": int(total_points - inside_count),
            "percentage_inside": float((inside_count / total_points) * 100 if total_points else 0),
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    for ply_path in sorted(results_dir.glob("*_yolo_gt.ply")):
        scene_id = ply_path.name.replace("_yolo_gt.ply", "")
        gt_path = results_dir / f"{scene_id}_gt_boxes.json"
        assignment_path = results_dir / f"{scene_id}_assignment.npy"
        if not gt_path.exists() or not assignment_path.exists():
            continue

        print(f"\nResults for scene {scene_id}")
        print(evaluate_scene(results_dir, scene_id).to_string(index=False))


if __name__ == "__main__":
    main()
