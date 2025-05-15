from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_calibration(calibration_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cam_to_velo_path = calibration_dir / "calib_cam_to_velo.txt"
    perspective_path = calibration_dir / "perspective.txt"

    values = list(map(float, cam_to_velo_path.read_text().strip().split()))
    cam_to_velo = np.eye(4)
    cam_to_velo[:3, :4] = np.array(values).reshape(3, 4)
    velo_to_cam = np.linalg.inv(cam_to_velo)

    for line in perspective_path.read_text().splitlines():
        if line.startswith("P_rect_00:"):
            projection_values = list(map(float, line.split(":", 1)[1].split()))
            return velo_to_cam, np.array(projection_values).reshape(3, 4)

    raise ValueError("P_rect_00 was not found in perspective.txt")


def load_lidar_points(bin_path: Path) -> np.ndarray:
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]


def project_points(points_3d: np.ndarray, projection: np.ndarray) -> np.ndarray:
    points_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_2d_hom = points_hom @ projection.T
    return points_2d_hom[:, :2] / points_2d_hom[:, 2:3]


def assign_points_to_masks(image_points: np.ndarray, masks: list[np.ndarray], image_shape: tuple[int, ...]) -> np.ndarray:
    assignment = -np.ones(len(image_points), dtype=int)
    if not masks:
        return assignment

    valid = (
        (image_points[:, 0] >= 0)
        & (image_points[:, 0] < image_shape[1])
        & (image_points[:, 1] >= 0)
        & (image_points[:, 1] < image_shape[0])
    )
    valid_points = image_points[valid].astype(int)

    for index, mask in enumerate(masks):
        resized = cv2.resize(mask, (image_shape[1], image_shape[0]))
        resized = (resized * 255).astype(np.uint8)
        eroded = cv2.erode(resized, np.ones((5, 5), np.uint8), iterations=2)
        inside = eroded[valid_points[:, 1], valid_points[:, 0]] > 127
        assignment[np.where(valid)[0][inside]] = index

    return assignment


def make_point_cloud(points: np.ndarray, assignment: np.ndarray, color_count: int) -> o3d.geometry.PointCloud:
    colors = np.ones((len(points), 3)) * 0.2
    if color_count > 0:
        car_colors = generate_colors(color_count)
        for index in range(color_count):
            colors[assignment == index] = car_colors[index]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def load_ground_truth_boxes(bbox_dir: Path, scene_id: str) -> list[dict]:
    bbox_path = bbox_dir / f"BBoxes_{int(scene_id)}.json"
    data = json.loads(bbox_path.read_text())
    return data if isinstance(data, list) else data.get("boxes", [])


def transform_boxes_to_lidar(boxes: list[dict], velo_to_cam: np.ndarray) -> list[list[list[float]]]:
    lidar_boxes = []
    cam_to_velo = np.linalg.inv(velo_to_cam)
    for box in boxes:
        corners = np.array(box["corners_cam0"])
        corners_hom = np.hstack((corners, np.ones((8, 1))))
        lidar_boxes.append((cam_to_velo @ corners_hom.T).T[:, :3].tolist())
    return lidar_boxes


def generate_colors(count: int) -> np.ndarray:
    import colorsys

    return np.array([colorsys.hls_to_rgb(index / count, 0.5, 0.9) for index in range(count)])
