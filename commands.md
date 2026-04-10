# D-FINE → CoreML Workflow

## 1) Convert PyTorch Model → CoreML

```bash
python export_dfine_coreml.py \
  --config configs/dfine/dfine_hgnetv2_x_coco.yml \
  --checkpoint weights/dfine_x_obj2coco.pth \
  --size 512 \
  --precision float16 \
  --output weights/dfine_x_obj2coco.mlpackage \
  --benchmark \
  --benchmark_runs 80 \
  --benchmark_compute_units CPU_AND_GPU,ALL,CPU_ONLY
```

---

## 2) Run Inference

### Folder of images

```bash
python infer_coreml.py \
  --input /Users/alpha/Downloads/sample-videos-master/ \
  --model weights/dfine_x.mlpackage \
  --output_dir outputs
```

### Video inference (compare model sizes)

```bash
# D-FINE X
python infer_coreml.py \
  --input /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_x_obj2coco.mlpackage \
  --input_size 512 \
  --output_dir outputs

# D-FINE L
python infer_coreml.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_l.mlpackage \
  --output_dir outputs

# D-FINE M
python infer_coreml.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_m.mlpackage \
  --output_dir outputs
```

### YOLO CoreML conversion (compare with D-FINE)

```bash
python infer_coreml.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model yolo26n.mlpackage \
  --output_dir outputs \
  --compute_units auto
```

### Parallel throughput test (same input, separate model instance per worker)

```bash
# 1 worker (baseline)
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 1 \
  --stats_interval 1.0

# 2 workers
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 2 \
  --stats_interval 1.0

# 4 workers
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 4 \
  --input_size 512 \
  --stats_interval 1.0
```

### Frame skipping (do not infer every frame)

```bash
# Infer every frame (default)
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 4 \
  --input_size 512 \
  --frame_stride 1 \
  --stats_interval 1.0

# Infer every 2nd frame
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 8 \
  --input_size 528 \
  --frame_stride 2 \
  --stats_interval 1.0

# Infer every 3rd frame
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 4 \
  --input_size 512 \
  --frame_stride 3 \
  --stats_interval 1.0
```

### Parallel throughput test (different input per worker)

```bash
# 2 workers, different sources
python run_parallel_rtsp_infer.py \
  --input_sources \
"/Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4,rtsp://192.168.1.137:8554/live" \
  --num_streams 2 \
  --stats_interval 1.0

# 4 workers, source list is reused round-robin if shorter than num_streams
python run_parallel_rtsp_infer.py \
  --input_sources \
"/Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4,rtsp://192.168.1.137:8554/live" \
  --num_streams 4 \
  --stats_interval 1.0
```

### Mixed compute-unit scheduling per worker

```bash
# Try splitting workers between GPU and NE paths
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 2 \
  --per_stream_compute_units cpu_and_gpu,cpu_and_ne \
  --stats_interval 1.0

# 4 workers, compute units reused round-robin
python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --num_streams 4 \
  --per_stream_compute_units cpu_and_gpu,cpu_and_ne \
  --stats_interval 1.0
```

```bash
# Optional: test with RTSP input instead
python run_parallel_rtsp_infer.py \
  --input_source "rtsp://192.168.1.137:8554/live" \
  --num_streams 4 \
  --stats_interval 1.0
```

Notes:
- `run_parallel_rtsp_infer.py` launches one process per worker, so each worker has its own CoreML model object instance.
- Max-throughput mode is enabled by default (`--no_render --no_save_video` in workers).
- Use `--input_sources` when you want a different source per worker.
- Use `--per_stream_compute_units` to test mixed hardware scheduling across workers.
- Use `--frame_stride N` to skip frames (`N=2` means infer every other frame).
- Per-stream FPS usually drops as workers increase, while total FPS (`[FPS][ALL] sum_infer`) should stay close to device capacity.

---

## 3) Benchmark on COCO-format Dataset

```bash
python benchmark_coreml_coco.py \
  --model weights/dfine_x_obj2coco.mlpackage \
  --annotations dataset/Company.coco/train/_annotations.coco.json \
  --images_dir dataset/Company.coco/train \
  --compute_units auto \
  --output_json outputs/company_coco_coreml_report.json \
  --output_csv outputs/company_coco_coreml_metrics.csv
```

---

## 4) Inference for Specific Classes Only (Person = class 0)

```bash
python infer_coreml.py \
  --input dataset/Company.coco/train \
  --model weights/dfine_x_obj2coco.mlpackage \
  --output_dir outputs/company_coco_dfine_x \
  --classes "[0]"
```
