from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

from lidar_utils import (
    assign_points_to_masks,
    load_calibration,
    load_ground_truth_boxes,
    load_lidar_points,
    make_point_cloud,
    project_points,
    transform_boxes_to_lidar,
)

CAR_CLASS_ID = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project KITTI-360 LiDAR points into YOLO car masks.")
    parser.add_argument("--dataset-root", default=os.getenv("KITTI360_PROJECT_ROOT", r"D:\M\Projects\KITTI-360_sample\Lidar-Project"))
    parser.add_argument("--model-path", default=os.getenv("YOLO_MODEL_PATH", r"D:\M\Projects\models\yolov8s-seg.pt"))
    parser.add_argument("--drive", default="2013_05_28_drive_0000_sync")
    parser.add_argument("--camera", default="image_00")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def get_car_masks(result) -> list[np.ndarray]:
    if result.masks is None:
        return []

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    return [masks[index] for index, class_id in enumerate(classes) if class_id == CAR_CLASS_ID]


def process_frame(
    lidar_path: Path,
    image_path: Path,
    bbox_dir: Path,
    output_dir: Path,
    velo_to_cam: np.ndarray,
    projection: np.ndarray,
    model: YOLO,
) -> None:
    scene_id = lidar_path.stem
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping {scene_id}: image not found")
        return

    lidar_points = load_lidar_points(lidar_path)
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    camera_points = lidar_hom @ velo_to_cam.T
    valid_depth = camera_points[:, 2] > 0
    camera_points = camera_points[valid_depth]
    lidar_points = lidar_points[valid_depth]

    image_points = project_points(camera_points[:, :3], projection)
    car_masks = get_car_masks(model(image)[0])
    assignment = assign_points_to_masks(image_points, car_masks, image.shape)
    point_cloud = make_point_cloud(lidar_points, assignment, len(car_masks))

    np.save(output_dir / f"{scene_id}_assignment.npy", assignment)
    o3d.io.write_point_cloud(str(output_dir / f"{scene_id}_yolo_gt.ply"), point_cloud)

    bbox_path = bbox_dir / f"BBoxes_{int(scene_id)}.json"
    if bbox_path.exists():
        boxes = load_ground_truth_boxes(bbox_dir, scene_id)
        lidar_boxes = transform_boxes_to_lidar(boxes, velo_to_cam)
        (output_dir / f"{scene_id}_gt_boxes.json").write_text(json.dumps(lidar_boxes, indent=2))

    print(f"Saved outputs for frame {scene_id}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "output_yolo_gt_ply"
    output_dir.mkdir(parents=True, exist_ok=True)

    lidar_dir = dataset_root / "data_3d_raw" / args.drive / "velodyne_points" / "data"
    image_dir = dataset_root / "data_2d_raw" / args.drive / args.camera / "data_rect"
    bbox_dir = dataset_root / "bboxes_3D_cam0"
    velo_to_cam, projection = load_calibration(dataset_root / "calibration")
    model = YOLO(args.model_path)

    for lidar_path in sorted(lidar_dir.glob("*.bin")):
        process_frame(
            lidar_path=lidar_path,
            image_path=image_dir / f"{lidar_path.stem}.png",
            bbox_dir=bbox_dir,
            output_dir=output_dir,
            velo_to_cam=velo_to_cam,
            projection=projection,
            model=model,
        )


if __name__ == "__main__":
    main()
