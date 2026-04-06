python dfine_to_coreml.py --config configs/dfine/dfine_hgnetv2_x_coco.yml --checkpoint weights/dfine_x_coco.pth --size 640 --precision float16 --output weights/dfine_x.mlpackage --benchmark --benchmark_runs 80 --benchmark_compute_units CPU_AND_GPU,ALL,CPU_ONLY

python run_inference.py --input /Users/alpha/Downloads/sample-videos-master/ --model weights/dfine_x.mlpackage --output_dir outputs

python run_inference.py --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 --model weights/dfine_x.mlpackage --output_dir outputs
python run_inference.py --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 --model weights/dfine_l.mlpackage --output_dir outputs
python run_inference.py --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 --model weights/dfine_m.mlpackage --output_dir outputs


python run_inference.py --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 --model yolo26n.mlpackage --output_dir outputs --compute_units auto
