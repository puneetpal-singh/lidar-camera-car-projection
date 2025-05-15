from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import json
import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a projected point cloud with filtered 3D boxes.")
    parser.add_argument("--results-dir", default=r"D:\M\Projects\KITTI-360_sample\Lidar-Project\output_yolo_gt_ply")
    parser.add_argument("--scene-id", default="0000002033")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    return parser.parse_args()


def create_bbox_lines(corners: np.ndarray, color: list[float]) -> o3d.geometry.LineSet:
    lines = [[0, 5], [1, 4], [2, 7], [3, 6], [0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([lighten_color(color)] * len(lines))
    return line_set


def lighten_color(color: list[float], factor: float = 0.7) -> list[float]:
    color_array = np.array(color)
    return np.clip(color_array * factor + np.ones(3) * (1 - factor), 0, 1).tolist()


def points_in_obb(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    center = np.mean(corners, axis=0)
    axes = np.stack([
        normalize(corners[1] - corners[0]),
        normalize(corners[3] - corners[1]),
        normalize(corners[4] - corners[0]),
    ], axis=1)
    local_points = (points - center) @ axes
    extents = np.array([
        np.linalg.norm(corners[1] - corners[0]),
        np.linalg.norm(corners[3] - corners[1]),
        np.linalg.norm(corners[4] - corners[0]),
    ]) / 2
    return np.all(np.abs(local_points) <= extents, axis=1)


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def obb_volume(corners: np.ndarray) -> float:
    return float(
        np.linalg.norm(corners[1] - corners[0])
        * np.linalg.norm(corners[3] - corners[1])
        * np.linalg.norm(corners[4] - corners[0])
    )


def aabb_iou(corners_a: np.ndarray, corners_b: np.ndarray) -> float:
    min_a, max_a = np.min(corners_a, axis=0), np.max(corners_a, axis=0)
    min_b, max_b = np.min(corners_b, axis=0), np.max(corners_b, axis=0)
    overlap = np.minimum(max_a, max_b) - np.maximum(min_a, min_b)
    if np.any(overlap <= 0):
        return 0.0
    intersection = float(np.prod(overlap))
    return intersection / (obb_volume(corners_a) + obb_volume(corners_b) - intersection)


def select_boxes(points: np.ndarray, colors: np.ndarray, assignment: np.ndarray, boxes: list, iou_threshold: float) -> list[dict]:
    boxes_by_color = defaultdict(list)
    for box in boxes:
        corners = np.array(box)
        inside_mask = points_in_obb(points, corners)
        matched_assignments = assignment[inside_mask]
        matched_assignments = matched_assignments[matched_assignments >= 0]
        if len(matched_assignments) == 0:
            continue

        car_id = np.bincount(matched_assignments).argmax()
        car_color = colors[assignment == car_id][0] if np.any(assignment == car_id) else np.zeros(3)
        boxes_by_color[tuple(car_color)].append({"corners": corners, "color": car_color.tolist(), "point_count": int(np.sum(inside_mask))})

    selected = []
    for box_list in boxes_by_color.values():
        kept = []
        for candidate in sorted(box_list, key=lambda item: item["point_count"], reverse=True):
            if all(aabb_iou(candidate["corners"], existing["corners"]) <= iou_threshold for existing in kept):
                kept.append(candidate)
        selected.extend(kept)
    return selected


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    point_cloud = o3d.io.read_point_cloud(str(results_dir / f"{args.scene_id}_yolo_gt.ply"))
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    assignment = np.load(results_dir / f"{args.scene_id}_assignment.npy")
    boxes = json.loads((results_dir / f"{args.scene_id}_gt_boxes.json").read_text())
    selected_boxes = select_boxes(points, colors, assignment, boxes, args.iou_threshold)
    box_lines = [create_bbox_lines(box["corners"], box["color"]) for box in selected_boxes]
    o3d.visualization.draw_geometries([point_cloud] + box_lines)


if __name__ == "__main__":
    main()
