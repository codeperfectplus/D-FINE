# D-FINE → CoreML Workflow

## 1) Convert PyTorch Model → CoreML

```bash
python dfine_to_coreml.py \
  --config configs/dfine/dfine_hgnetv2_x_coco.yml \
  --checkpoint weights/dfine_x_obj2coco.pth \
  --size 640 \
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
python run_inference.py \
  --input /Users/alpha/Downloads/sample-videos-master/ \
  --model weights/dfine_x.mlpackage \
  --output_dir outputs
```

### Video inference (compare model sizes)

```bash
# D-FINE X
python run_inference.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_x.mlpackage \
  --output_dir outputs

# D-FINE L
python run_inference.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_l.mlpackage \
  --output_dir outputs

# D-FINE M
python run_inference.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model weights/dfine_m.mlpackage \
  --output_dir outputs
```

### YOLO CoreML conversion (compare with D-FINE)

```bash
python run_inference.py \
  --video /Users/alpha/codespace/experiments/human_detection/input/people-detection.mp4 \
  --model yolo26n.mlpackage \
  --output_dir outputs \
  --compute_units auto
```

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
python run_inference.py \
  --input dataset/Company.coco/train \
  --model weights/dfine_x_obj2coco.mlpackage \
  --output_dir outputs/company_coco_dfine_x \
  --classes "[0]"
```
