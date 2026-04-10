python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_x_obj2coco.mlpackage \
  --num_streams 1 \
  --input_size 640 \
  --frame_stride 2 \
  --compute_units cpu_and_gpu

Reports parsed: 8
Frames read (all streams): 4768
Frames inferred (all streams): 2384
Frames skipped (all streams): 2384
Average effective inference FPS (frame-weighted): 3.30
Average model-only inference FPS (latency-weighted): 3.33
Overall throughput FPS (wall-clock, stagger-aware): 24.57 over 97.04s
Peak parallel inferred FPS (window sum): 28.32
Average parallel inferred FPS (active window): 26.07
Peak parallel model FPS (window sum): 28.57
Average parallel model FPS (active window): 26.28

python run_parallel_rtsp_infer.py \
  --input_source /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/rf-detr-xxlarge.mlpackage \
  --num_streams 8 \
  --input_size 520 \
  --frame_stride 2 \
  --compute_units cpu_and_ne


Aggregate inference summary
Reports parsed: 8
Frames read (all streams): 4768
Frames inferred (all streams): 2384
Frames skipped (all streams): 2384
Average effective inference FPS (frame-weighted): 5.99
Average model-only inference FPS (latency-weighted): 6.09
Overall throughput FPS (wall-clock, stagger-aware): 24.31 over 98.05s
Peak parallel inferred FPS (window sum): 66.21
Average parallel inferred FPS (active window): 35.91
Peak parallel model FPS (window sum): 68.61
Average parallel model FPS (active window): 36.55