import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
import coremltools as ct

from coreml_predictor import DFineCoreMLPredictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
COMPUTE_UNIT_MAP = {
    "all": ct.ComputeUnit.ALL,
    "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
    "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    "cpu_only": ct.ComputeUnit.CPU_ONLY,
}


def is_stream_uri(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def sanitize_source_name(value: str) -> str:
    sanitized = []
    for ch in value.lower():
        if ch.isalnum():
            sanitized.append(ch)
        elif ch in {"-", "_"}:
            sanitized.append(ch)
        else:
            sanitized.append("_")

    collapsed = "".join(sanitized).strip("_")
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed[:80] or "stream"


def parse_class_ids_arg(classes_arg: str) -> Optional[Set[int]]:
    value = classes_arg.strip()
    if value.lower() in {"all", "*"}:
        return None

    value = value.strip("[]")
    if not value:
        return None

    class_ids: Set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            class_ids.add(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid class id '{token}' in --classes. "
                "Use 'all' or comma-separated integers like '0,1'."
            ) from exc

    return class_ids or None


def filter_detections_by_class(detections, class_ids: Optional[Set[int]]):
    if class_ids is None:
        return detections
    return [det for det in detections if int(det.get("class_id", -1)) in class_ids]


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


def get_stream_output_paths(stream_source: str, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_source_name(stream_source)
    output_media = output_dir / f"{stem}_coreml_output.mp4"
    output_json = output_dir / f"{stem}_coreml_report.json"
    return ensure_unique_output_pair(output_media, output_json)


def run_video_inference(
    predictor: DFineCoreMLPredictor,
    model_path: Path,
    video_path: str | Path,
    output_video: Path,
    output_json: Path,
    class_ids: Optional[Set[int]] = None,
    stream_label: Optional[str] = None,
    fps_log_interval: float = 2.0,
    model_instance: Optional[str] = None,
    render_output: bool = True,
    save_output_video: bool = True,
    frame_stride: int = 1,
):
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if src_fps <= 0:
        src_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if save_output_video:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, src_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to create output video: {output_video}")

    latencies = []
    frame_read_count = 0
    frame_inferred_count = 0
    stream_name = stream_label or str(video_path)
    model_instance_name = model_instance or "default"
    total_start = time.perf_counter()
    log_window_start = time.perf_counter()
    window_read_frames = 0
    window_infer_frames = 0
    window_infer_ms = 0.0

    print(f"Running inference on: {video_path}")
    print(f"Model instance: {model_instance_name}")
    print(f"Frame stride: {frame_stride}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_read_count += 1
        window_read_frames += 1
        timestamp_s = (frame_read_count - 1) / src_fps

        should_infer = ((frame_read_count - 1) % frame_stride) == 0
        detections = []
        latency_ms = 0.0
        infer_fps = 0.0

        if should_infer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections, latency_ms = predictor.predict(rgb, orig_size=(height, width))
            detections = filter_detections_by_class(detections, class_ids)
            infer_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            latencies.append(latency_ms)
            frame_inferred_count += 1
            window_infer_frames += 1
            window_infer_ms += latency_ms

        if render_output:
            if should_infer:
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
            if should_infer:
                cv2.putText(
                    frame,
                    f"lat={latency_ms:.1f}ms fps={infer_fps:.1f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
            elif frame_stride > 1:
                cv2.putText(
                    frame,
                    f"skipped (stride={frame_stride})",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        if writer is not None:
            writer.write(frame)

        if fps_log_interval > 0:
            elapsed_s = time.perf_counter() - log_window_start
            if elapsed_s >= fps_log_interval and window_read_frames > 0:
                camera_fps = window_read_frames / elapsed_s
                effective_infer_fps = window_infer_frames / elapsed_s
                model_infer_fps = (
                    (window_infer_frames * 1000.0) / window_infer_ms
                    if window_infer_ms > 0
                    else 0.0
                )
                print(
                    "FPS_LOG "
                    + json.dumps(
                        {
                            "stream": stream_name,
                            "model_instance": model_instance_name,
                            "camera_fps": round(camera_fps, 3),
                            "infer_fps": round(effective_infer_fps, 3),
                            "model_infer_fps": round(model_infer_fps, 3),
                            "frames_total": frame_read_count,
                            "frames_inferred": frame_inferred_count,
                            "source_fps_nominal": round(src_fps, 3),
                            "frame_stride": frame_stride,
                        }
                    ),
                    flush=True,
                )
                log_window_start = time.perf_counter()
                window_read_frames = 0
                window_infer_frames = 0
                window_infer_ms = 0.0

    cap.release()
    if writer is not None:
        writer.release()

    total_elapsed_s = max(time.perf_counter() - total_start, 1e-9)
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    med_latency = statistics.median(latencies) if latencies else 0.0
    avg_infer_fps_model = (1000.0 / avg_latency) if avg_latency > 0 else 0.0
    effective_camera_fps = frame_read_count / total_elapsed_s
    effective_infer_fps = frame_inferred_count / total_elapsed_s

    report = {
        "input_type": "video",
        "input_path": str(video_path),
        "video_path": str(video_path),
        "model_path": str(model_path),
        "model_instance": model_instance_name,
        "frame_stride": frame_stride,
        "class_filter_ids": sorted(class_ids) if class_ids is not None else "all",
        "frames_read": frame_read_count,
        "frames_inferred": frame_inferred_count,
        "frames_skipped": frame_read_count - frame_inferred_count,
        "frames_processed": frame_inferred_count,
        "source_video_fps": round(src_fps, 3),
        "effective_camera_fps": round(effective_camera_fps, 3),
        "effective_inference_fps": round(effective_infer_fps, 3),
        "output_video_saved": bool(save_output_video),
        "average_latency_ms": round(avg_latency, 3),
        "median_latency_ms": round(med_latency, 3),
        "average_inference_fps": round(avg_infer_fps_model, 3),
    }

    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def run_image_inference(
    predictor: DFineCoreMLPredictor,
    model_path: Path,
    image_path: Path,
    output_image: Path,
    output_json: Path,
    class_ids: Optional[Set[int]] = None,
):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to open image: {image_path}")

    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections, latency_ms = predictor.predict(rgb, orig_size=(height, width))
    detections = filter_detections_by_class(detections, class_ids)
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
        "class_filter_ids": sorted(class_ids) if class_ids is not None else "all",
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
    parser.add_argument(
        "--classes",
        default="all",
        help="Class IDs to keep. Use 'all' (default), '0', '0,1', or '[0,1]'.",
    )
    parser.add_argument(
        "--stream_label",
        default="",
        help="Optional stream label used in periodic FPS logs.",
    )
    parser.add_argument(
        "--fps_log_interval",
        type=float,
        default=2.0,
        help="Seconds between periodic FPS log lines for video/stream inputs.",
    )
    parser.add_argument(
        "--model_instance",
        default="",
        help="Optional label to identify the model instance in logs/reports.",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Disable drawing overlays for higher throughput.",
    )
    parser.add_argument(
        "--no_save_video",
        action="store_true",
        help="Skip writing output video for higher throughput (JSON report is still saved).",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Run inference every Nth frame (1 = infer every frame, 2 = every other frame).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_value = args.input or args.video
    assert input_value is not None
    use_stream_input = is_stream_uri(input_value)
    input_path = None if use_stream_input else Path(input_value)
    model_path = Path(args.model)
    output_dir = Path(args.output_dir)

    if not use_stream_input:
        assert input_path is not None
        if not input_path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {model_path}")

    exclude_dir = None
    if (
        input_path is not None
        and input_path.is_dir()
        and is_relative_to(output_dir.resolve(), input_path.resolve())
    ):
        exclude_dir = output_dir.resolve()

    if use_stream_input:
        print("Found 1 stream input.")
        stream_source = input_value
        input_files = []
    else:
        assert input_path is not None
        input_files = collect_media_inputs(input_path, exclude_dir=exclude_dir)
        print(f"Found {len(input_files)} supported input file(s).")
        stream_source = ""

    compute_units, compute_units_name = resolve_compute_units(
        args.compute_units,
        model_path,
    )
    print(f"Compute units: {compute_units_name}")

    class_ids = parse_class_ids_arg(args.classes)
    if class_ids is None:
        print("Class filter: all")
    else:
        print(f"Class filter: {sorted(class_ids)}")

    predictor = DFineCoreMLPredictor(
        str(model_path),
        conf_threshold=args.conf_threshold,
        input_size=args.input_size,
        compute_units=compute_units,
    )

    input_root = input_path if (input_path is not None and input_path.is_dir()) else None
    reports = []

    if use_stream_input:
        output_media, output_json = get_stream_output_paths(stream_source, output_dir)
        print(f"[1/1] Processing video stream: {stream_source}")
        report = run_video_inference(
            predictor=predictor,
            model_path=model_path,
            video_path=stream_source,
            output_video=output_media,
            output_json=output_json,
            class_ids=class_ids,
            stream_label=args.stream_label or None,
            fps_log_interval=args.fps_log_interval,
            model_instance=args.model_instance or None,
            render_output=not args.no_render,
            save_output_video=not args.no_save_video,
            frame_stride=args.frame_stride,
        )
        report["output_path"] = str(output_media)
        report["report_path"] = str(output_json)
        reports.append(report)
        if not args.no_save_video:
            print(f"Output saved : {output_media}")
        else:
            print("Output saved : disabled (--no_save_video)")
        print(f"JSON report  : {output_json}")
    else:
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
                    class_ids=class_ids,
                    stream_label=args.stream_label or None,
                    fps_log_interval=args.fps_log_interval,
                    model_instance=args.model_instance or None,
                    render_output=not args.no_render,
                    save_output_video=not args.no_save_video,
                    frame_stride=args.frame_stride,
                )
            elif media_kind == "image":
                report = run_image_inference(
                    predictor=predictor,
                    model_path=model_path,
                    image_path=media_path,
                    output_image=output_media,
                    output_json=output_json,
                    class_ids=class_ids,
                )
            else:
                continue

            report["output_path"] = str(output_media)
            report["report_path"] = str(output_json)
            reports.append(report)

            if not args.no_save_video:
                print(f"Output saved : {output_media}")
            else:
                print("Output saved : disabled (--no_save_video)")
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