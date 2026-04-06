import argparse
import json
import statistics
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import coremltools as ct

from dfine_coreml_infer import DFineCoreMLPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
COMPUTE_UNIT_MAP = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box_xyxy"]]
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            2,
        )


def get_media_kind(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def infer_model_family(model_path: Path) -> str:
    name = model_path.name.lower()
    if "yolo" in name:
        return "yolo"
    return "dfine"


def resolve_compute_units(choice: str, model_path: Path):
    family = infer_model_family(model_path)

    if family == "yolo" and choice == "cpu_and_gpu":
        print(
            "Warning: YOLO CoreML with cpu_and_gpu can abort on this system "
            "(MPSGraph MLIR failure). Forcing compute units to cpu_and_ne."
        )
        return ct.ComputeUnit.CPU_AND_NE, "cpu_and_ne"

    if choice != "auto":
        return COMPUTE_UNIT_MAP[choice], choice

    if family == "yolo":
        # YOLO CoreML packages in this repo can crash on GPU backend, while
        # CPU_AND_NE and ALL are stable and much faster than CPU_ONLY.
        return ct.ComputeUnit.CPU_AND_NE, "cpu_and_ne"
    return ct.ComputeUnit.CPU_AND_GPU, "cpu_and_gpu"


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def collect_media_inputs(input_path: Path, exclude_dir: Optional[Path] = None) -> List[Path]:
    if input_path.is_file():
        media_kind = get_media_kind(input_path)
        if media_kind is None:
            raise ValueError(f"Unsupported file type: {input_path}")
        return [input_path]

    exclude_dir_resolved = exclude_dir.resolve() if exclude_dir else None
    media_files: List[Path] = []
    for path in input_path.rglob("*"):
        if not path.is_file():
            continue
        if exclude_dir_resolved and is_relative_to(path.resolve(), exclude_dir_resolved):
            continue
        if get_media_kind(path):
            media_files.append(path)

    media_files.sort(key=lambda p: str(p))
    if not media_files:
        raise ValueError(f"No supported image/video files found in: {input_path}")
    return media_files


def ensure_unique_output_pair(output_media: Path, output_json: Path) -> Tuple[Path, Path]:
    if not output_media.exists() and not output_json.exists():
        return output_media, output_json

    index = 1
    while True:
        media_candidate = output_media.with_name(
            f"{output_media.stem}_{index}{output_media.suffix}"
        )
        json_candidate = output_json.with_name(
            f"{output_json.stem}_{index}{output_json.suffix}"
        )
        if not media_candidate.exists() and not json_candidate.exists():
            return media_candidate, json_candidate
        index += 1


def get_output_paths(
    input_path: Path,
    output_dir: Path,
    input_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    media_kind = get_media_kind(input_path)
    if media_kind is None:
        raise ValueError(f"Unsupported file type: {input_path}")

    target_dir = output_dir
    if input_root and input_root.is_dir():
        try:
            rel_parent = input_path.parent.relative_to(input_root)
        except ValueError:
            rel_parent = Path()
        target_dir = output_dir / rel_parent

    target_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    if media_kind == "video":
        output_media = target_dir / f"{stem}_coreml_output.mp4"
    else:
        image_suffix = input_path.suffix.lower()
        if image_suffix not in IMAGE_EXTENSIONS:
            image_suffix = ".jpg"
        output_media = target_dir / f"{stem}_coreml_output{image_suffix}"
    output_json = target_dir / f"{stem}_coreml_report.json"

    return ensure_unique_output_pair(output_media, output_json)


def run_video_inference(
    predictor: DFineCoreMLPredictor,
    model_path: Path,
    video_path: Path,
    output_video: Path,
    output_json: Path,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if src_fps <= 0:
        src_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, src_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video: {output_video}")

    latencies = []
    frame_index = 0

    print(f"Running inference on: {video_path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_s = frame_index / src_fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections, latency_ms = predictor.predict(rgb, orig_size=(height, width))
        infer_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        latencies.append(latency_ms)
        draw_detections(frame, detections)

        cv2.putText(
            frame,
            f"t={timestamp_s:.2f}s",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            f"lat={latency_ms:.1f}ms fps={infer_fps:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        writer.write(frame)

        frame_index += 1

    cap.release()
    writer.release()

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    med_latency = statistics.median(latencies) if latencies else 0.0
    avg_infer_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0

    report = {
        "input_type": "video",
        "input_path": str(video_path),
        "video_path": str(video_path),
        "model_path": str(model_path),
        "frames_processed": frame_index,
        "source_video_fps": round(src_fps, 3),
        "average_latency_ms": round(avg_latency, 3),
        "median_latency_ms": round(med_latency, 3),
        "average_inference_fps": round(avg_infer_fps, 3),
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def run_image_inference(
    predictor: DFineCoreMLPredictor,
    model_path: Path,
    image_path: Path,
    output_image: Path,
    output_json: Path,
):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to open image: {image_path}")

    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections, latency_ms = predictor.predict(rgb, orig_size=(height, width))
    infer_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

    draw_detections(frame, detections)
    cv2.putText(
        frame,
        f"lat={latency_ms:.1f}ms fps={infer_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    if not cv2.imwrite(str(output_image), frame):
        raise RuntimeError(f"Failed to write output image: {output_image}")

    report = {
        "input_type": "image",
        "input_path": str(image_path),
        "image_path": str(image_path),
        "model_path": str(model_path),
        "image_width": width,
        "image_height": height,
        "detections_count": len(detections),
        "latency_ms": round(latency_ms, 3),
        "inference_fps": round(infer_fps, 3),
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run D-FINE CoreML inference on image/video file(s)"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input",
        help="Path to input image/video file or directory",
    )
    source_group.add_argument(
        "--video",
        help="Deprecated alias for --input (video file path)",
    )
    parser.add_argument(
        "--model",
        default="dfine_x.mlpackage",
        help="Path to CoreML .mlpackage",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to save output video and reports",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.4,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=640,
        help="CoreML model input size used at conversion time",
    )
    parser.add_argument(
        "--compute_units",
        choices=["auto", "all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"],
        default="auto",
        help="CoreML compute units. 'auto' uses CPU_AND_NE for YOLO models and CPU_AND_GPU otherwise.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input or args.video)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {model_path}")

    exclude_dir = None
    if input_path.is_dir() and is_relative_to(output_dir.resolve(), input_path.resolve()):
        exclude_dir = output_dir.resolve()

    input_files = collect_media_inputs(input_path, exclude_dir=exclude_dir)
    print(f"Found {len(input_files)} supported input file(s).")

    compute_units, compute_units_name = resolve_compute_units(
        args.compute_units,
        model_path,
    )
    print(f"Compute units: {compute_units_name}")

    predictor = DFineCoreMLPredictor(
        str(model_path),
        conf_threshold=args.conf_threshold,
        input_size=args.input_size,
        compute_units=compute_units,
    )

    input_root = input_path if input_path.is_dir() else None
    reports = []

    for index, media_path in enumerate(input_files, start=1):
        media_kind = get_media_kind(media_path)
        output_media, output_json = get_output_paths(
            media_path,
            output_dir,
            input_root=input_root,
        )
        print(f"[{index}/{len(input_files)}] Processing {media_kind}: {media_path}")

        if media_kind == "video":
            report = run_video_inference(
                predictor=predictor,
                model_path=model_path,
                video_path=media_path,
                output_video=output_media,
                output_json=output_json,
            )
        elif media_kind == "image":
            report = run_image_inference(
                predictor=predictor,
                model_path=model_path,
                image_path=media_path,
                output_image=output_media,
                output_json=output_json,
            )
        else:
            continue

        report["output_path"] = str(output_media)
        report["report_path"] = str(output_json)
        reports.append(report)

        print(f"Output saved : {output_media}")
        print(f"JSON report  : {output_json}")

    if not reports:
        raise RuntimeError("No inputs were processed.")

    if len(reports) > 1:
        output_dir.mkdir(parents=True, exist_ok=True)
        batch_report_path = output_dir / "coreml_batch_report.json"
        batch_report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"Batch report : {batch_report_path}")

    print("\nInference complete.")
    print(f"Inputs       : {len(reports)}")

    if len(reports) == 1:
        report = reports[0]
        print(f"JSON report  : {report['report_path']}")
        if report["input_type"] == "video":
            print(f"Frames       : {report['frames_processed']}")
            print(f"Avg latency  : {report['average_latency_ms']:.2f} ms")
            print(f"Avg infer FPS: {report['average_inference_fps']:.2f}")
        else:
            print(f"Detections   : {report['detections_count']}")
            print(f"Latency      : {report['latency_ms']:.2f} ms")
            print(f"Infer FPS    : {report['inference_fps']:.2f}")
    else:
        latency_values = [
            report["average_latency_ms"]
            if report["input_type"] == "video"
            else report["latency_ms"]
            for report in reports
        ]
        avg_latency = statistics.mean(latency_values)
        avg_fps = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
        print(f"Avg latency  : {avg_latency:.2f} ms")
        print(f"Avg infer FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()