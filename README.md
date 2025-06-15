# LiDAR Camera Car Projection

This project projects KITTI-360 LiDAR points into camera images, uses YOLO segmentation masks to isolate car points, exports colored point clouds, and evaluates detected car point clusters against 3D bounding boxes.

## What It Does

- Loads KITTI-360 LiDAR frames, camera frames, and calibration files.
- Projects LiDAR points into the camera plane.
- Runs YOLO instance segmentation for cars.
- Assigns projected LiDAR points to car masks.
- Exports colored `.ply` point clouds and matching metadata.
- Evaluates projected clusters against available 3D ground-truth boxes.
- Visualizes final point clouds with 3D box overlays.

## Data

The dataset is not included in this repository. Place the KITTI-360-style project data somewhere local, for example:

```powershell
D:\M\Projects\KITTI-360_sample\Lidar-Project
```

Expected folders under the dataset root:

- `data_3d_raw`
- `data_2d_raw`
- `calibration`
- `bboxes_3D_cam0`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download a YOLO segmentation model such as `yolov8s-seg.pt` or `yolov8x-seg.pt` and keep it outside the repository, for example:

```powershell
D:\M\Projects\models\yolov8s-seg.pt
```

## Run Projection

```powershell
python project_yolo_lidar.py --dataset-root "D:\M\Projects\KITTI-360_sample\Lidar-Project" --model-path "D:\M\Projects\models\yolov8s-seg.pt"
```

## Evaluate

```powershell
python evaluate_3d_detection.py --results-dir "D:\M\Projects\KITTI-360_sample\Lidar-Project\output_yolo_gt_ply"
```

## View A Frame

```powershell
python view_pointcloud.py --results-dir "D:\M\Projects\KITTI-360_sample\Lidar-Project\output_yolo_gt_ply" --scene-id 0000002033
```

Generated outputs, dataset files, model weights, and copied external toolkits are intentionally excluded from Git.


## Updates
- Switched to YOLOv8x for improved segmentation.
