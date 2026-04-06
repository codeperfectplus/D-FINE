import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import coremltools as ct

from dfine_coreml_infer import DFineCoreMLPredictor


COMPUTE_UNIT_MAP = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def resolve_compute_units(choice: str, model_path: Path):
    name = model_path.name.lower()
    if choice == "auto":
        if "yolo" in name:
            return ct.ComputeUnit.CPU_AND_NE, "cpu_and_ne"
        return ct.ComputeUnit.CPU_AND_GPU, "cpu_and_gpu"

    if "yolo" in name and choice == "cpu_and_gpu":
        print(
            "Warning: YOLO CoreML with cpu_and_gpu can abort on this system. "
            "Using cpu_and_ne instead."
        )
        return ct.ComputeUnit.CPU_AND_NE, "cpu_and_ne"

    return COMPUTE_UNIT_MAP[choice], choice


def xywh_to_xyxy(box_xywh: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box_xywh]
    return [x, y, x + w, y + h]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_image(
    pred_boxes: List[Sequence[float]],
    gt_boxes: List[Sequence[float]],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    matched_gt: Set[int] = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            current_iou = iou_xyxy(pred, gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt = gt_idx

        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def compute_ap50(
    all_predictions: List[Tuple[int, float, Sequence[float]]],
    gt_by_image: Dict[int, List[Sequence[float]]],
    iou_threshold: float,
) -> float:
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt == 0:
        return 0.0

    predictions = sorted(all_predictions, key=lambda x: x[1], reverse=True)
    matched_by_image: Dict[int, Set[int]] = {image_id: set() for image_id in gt_by_image}

    tp_flags: List[int] = []
    fp_flags: List[int] = []

    for image_id, _, pred_box in predictions:
        gt_boxes = gt_by_image.get(image_id, [])
        matched_gt = matched_by_image.setdefault(image_id, set())

        best_iou = 0.0
        best_gt = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            current_iou = iou_xyxy(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt = gt_idx

        if best_gt >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_gt)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    cumulative_tp: List[int] = []
    cumulative_fp: List[int] = []
    running_tp = 0
    running_fp = 0
    for tp, fp in zip(tp_flags, fp_flags):
        running_tp += tp
        running_fp += fp
        cumulative_tp.append(running_tp)
        cumulative_fp.append(running_fp)

    recalls: List[float] = []
    precisions: List[float] = []
    for ctp, cfp in zip(cumulative_tp, cumulative_fp):
        recall = ctp / total_gt
        precision = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
        recalls.append(recall)
        precisions.append(precision)

    ap = 0.0
    for r in [i / 100.0 for i in range(101)]:
        precision_at_r = 0.0
        for rec, prec in zip(recalls, precisions):
            if rec >= r and prec > precision_at_r:
                precision_at_r = prec
        ap += precision_at_r
    return ap / 101.0


def choose_gt_category_ids(categories: List[Dict]) -> Set[int]:
    selected: Set[int] = set()
    for cat in categories:
        name = str(cat.get("name", "")).lower()
        supercat = str(cat.get("supercategory", "")).lower()
        if "person" in name or "human" in name or "person" in supercat or "human" in supercat:
            selected.add(int(cat["id"]))

    if selected:
        return selected

    return {int(cat["id"]) for cat in categories}


def parse_class_ids(text: str) -> Set[int]:
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CoreML model on COCO-format dataset")
    parser.add_argument(
        "--model",
        default="weights/dfine_x_obj2coco.mlpackage",
        help="Path to CoreML .mlpackage",
    )
    parser.add_argument(
        "--annotations",
        default="dataset/Company.coco/train/_annotations.coco.json",
        help="Path to COCO annotations JSON",
    )
    parser.add_argument(
        "--images_dir",
        default=None,
        help="Directory containing images (default: parent folder of annotations)",
    )
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument(
        "--compute_units",
        choices=["auto", "all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"],
        default="auto",
    )
    parser.add_argument(
        "--pred_class_ids",
        default="0",
        help="Comma-separated predicted class IDs treated as Human (default: 0)",
    )
    parser.add_argument(
        "--output_json",
        default="outputs/company_coco_coreml_report.json",
    )
    parser.add_argument(
        "--output_csv",
        default="outputs/company_coco_coreml_metrics.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir) if args.images_dir else annotations_path.parent
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    with annotations_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    gt_category_ids = choose_gt_category_ids(categories)
    gt_category_names = [
        c.get("name", str(c.get("id")))
        for c in categories
        if int(c.get("id", -1)) in gt_category_ids
    ]
    pred_class_ids = parse_class_ids(args.pred_class_ids)

    print(f"Images in annotations: {len(coco.get('images', []))}")
    print(f"GT categories merged as Human: {sorted(gt_category_ids)}")
    print(f"GT category names: {gt_category_names}")
    print(f"Predicted class IDs treated as Human: {sorted(pred_class_ids)}")

    anns_by_image: Dict[int, List[List[float]]] = {}
    for ann in coco.get("annotations", []):
        if int(ann.get("category_id", -1)) not in gt_category_ids:
            continue
        image_id = int(ann["image_id"])
        anns_by_image.setdefault(image_id, []).append(xywh_to_xyxy(ann["bbox"]))

    compute_units, compute_units_name = resolve_compute_units(args.compute_units, model_path)
    print(f"Compute units: {compute_units_name}")

    predictor = DFineCoreMLPredictor(
        str(model_path),
        conf_threshold=args.conf_threshold,
        input_size=args.input_size,
        compute_units=compute_units,
    )

    per_image_rows: List[Dict] = []
    all_predictions: List[Tuple[int, float, Sequence[float]]] = []
    latencies_ms: List[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    missing_images = 0

    images = coco.get("images", [])
    for idx, image in enumerate(images, start=1):
        image_id = int(image["id"])
        file_name = image["file_name"]
        image_path = images_dir / file_name

        gt_boxes = anns_by_image.get(image_id, [])
        if not image_path.exists():
            missing_images += 1
            per_image_rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "gt_count": len(gt_boxes),
                    "pred_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": len(gt_boxes),
                    "latency_ms": "",
                    "missing_image": 1,
                }
            )
            total_fn += len(gt_boxes)
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            missing_images += 1
            per_image_rows.append(
                {
                    "image_id": image_id,
                    "file_name": file_name,
                    "gt_count": len(gt_boxes),
                    "pred_count": 0,
                    "tp": 0,
                    "fp": 0,
                    "fn": len(gt_boxes),
                    "latency_ms": "",
                    "missing_image": 1,
                }
            )
            total_fn += len(gt_boxes)
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections, latency_ms = predictor.predict(rgb, orig_size=(h, w))
        latencies_ms.append(latency_ms)

        selected_dets = [
            det
            for det in detections
            if int(det["class_id"]) in pred_class_ids and float(det["score"]) >= args.conf_threshold
        ]
        selected_dets.sort(key=lambda d: float(d["score"]), reverse=True)

        pred_boxes = [det["box_xyxy"] for det in selected_dets]
        tp, fp, fn = match_image(pred_boxes, gt_boxes, args.iou_threshold)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        for det in selected_dets:
            all_predictions.append((image_id, float(det["score"]), det["box_xyxy"]))

        per_image_rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "gt_count": len(gt_boxes),
                "pred_count": len(selected_dets),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "latency_ms": round(latency_ms, 4),
                "missing_image": 0,
            }
        )

        if idx % 100 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)} images")

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    ap50 = compute_ap50(all_predictions, anns_by_image, args.iou_threshold)

    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
    med_latency = statistics.median(latencies_ms) if latencies_ms else 0.0
    infer_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

    report = {
        "model_path": str(model_path),
        "annotations_path": str(annotations_path),
        "images_dir": str(images_dir),
        "images_total": len(images),
        "images_missing": missing_images,
        "images_processed": len(images) - missing_images,
        "gt_category_ids_merged_to_human": sorted(gt_category_ids),
        "gt_category_names_merged_to_human": gt_category_names,
        "predicted_class_ids_treated_as_human": sorted(pred_class_ids),
        "conf_threshold": args.conf_threshold,
        "iou_threshold": args.iou_threshold,
        "compute_units": compute_units_name,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "AP50": round(ap50, 6),
        "Precision@IoU0.5": round(precision, 6),
        "Recall@IoU0.5": round(recall, 6),
        "AVG Inference Time (ms/frame)": round(avg_latency, 4),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "ap50": round(ap50, 6),
        "average_latency_ms": round(avg_latency, 4),
        "median_latency_ms": round(med_latency, 4),
        "average_inference_fps": round(infer_fps, 4),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "file_name",
                "gt_count",
                "pred_count",
                "tp",
                "fp",
                "fn",
                "latency_ms",
                "missing_image",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    print("\nBenchmark complete.")
    print(f"Report JSON : {output_json}")
    print(f"Per-image CSV: {output_csv}")
    print(f"AP50                : {report['AP50']:.4f}")
    print(f"Precision@IoU0.5    : {report['Precision@IoU0.5']:.4f}")
    print(f"Recall@IoU0.5       : {report['Recall@IoU0.5']:.4f}")
    print(f"AVG Inference Time  : {report['AVG Inference Time (ms/frame)']:.2f} ms/frame")
    print(f"F1          : {report['f1']:.4f}")
    print(f"Avg latency : {report['average_latency_ms']:.2f} ms")
    print(f"Avg infer FPS: {report['average_inference_fps']:.2f}")


if __name__ == "__main__":
    main()