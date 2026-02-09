# Object Permanence

This repo includes a YOLOv8 video detection pipeline that samples frames, runs inference, and saves JSON results.

## Quickstart

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Run a single-video detection from the repo root:

```bash
python3 src/detection/run_yolo.py --video data/raw_videos/3sec_Left_to_Right.mp4 --sample-rate 5 --model yolov8n.pt
```

3. Or run batch detection over a directory:

```bash
python3 src/detection/run_yolo_batch.py --video-dir data/raw_videos --pattern "*.mp4" --sample-rate 5 --model yolov8n.pt
```

## Output

Detections are saved to:

```
experiments/results/detections/[video_name]_detections.json
```

Each entry includes the frame number and an array of detections with class id, class name, bounding box coordinates (xyxy), and confidence.

## Structure

- `src/detection/types.py`: Shared dataclasses for detections
- `src/detection/io.py`: Output path utilities and JSON writer
- `src/detection/sampler.py`: Frame sampling from video
- `src/detection/yolo_runner.py`: YOLOv8 inference wrapper
- `src/detection/pipeline.py`: Pipeline orchestration
- `src/detection/cli.py`: Shared CLI argument parsing and dispatch
- `src/detection/run_yolo.py`: Single-video CLI entry point
- `src/detection/run_yolo_batch.py`: Batch CLI entry point

## Config

- `--video`: input video path (single mode)
- `--video-dir`: input directory (batch mode)
- `--pattern`: glob for batch videos (default `*.mp4`)
- `--sample-rate`: sample every N frames (default 5)
- `--model`: YOLOv8 model name or path (default `yolov8n.pt`)
