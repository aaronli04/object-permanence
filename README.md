# Object Permanence

This repo includes a YOLOv8 video detection pipeline that samples frames, runs inference, and saves JSON results.

## Quickstart

1. Install dependencies (at minimum `opencv-python` and `ultralytics`).
2. Run the detection script from the repo root:

```bash
python src/detection/run_yolo.py --video data/raw_videos/3sec_Left_to_Right.mp4 --sample-rate 30 --model yolov8n.pt
```

## Output

Detections are saved to:

```
experiments/results/detections/[video_name]_detections.json
```

Each entry includes the frame number and an array of detections with class id, class name, bounding box coordinates (xyxy), and confidence.

## Structure

- `src/detection/sampler.py`: Frame sampling from video
- `src/detection/yolo_runner.py`: YOLOv8 inference wrapper
- `src/detection/output_writer.py`: JSON output writer
- `src/detection/pipeline.py`: Pipeline wiring and output path
- `src/detection/run_yolo.py`: CLI entry point

## Config

- `--video`: input video path
- `--sample-rate`: sample every N frames (default 30)
- `--model`: YOLOv8 model name or path (default `yolov8n.pt`)
