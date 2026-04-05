import argparse
import csv
import json
import statistics
from pathlib import Path

import cv2
import coremltools as ct

from dfine_coreml_infer import DFineCoreMLPredictor


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


def get_output_paths(video_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    output_video = output_dir / f"{stem}_coreml_output.mp4"
    output_json = output_dir / f"{stem}_coreml_report.json"
    return output_video, output_json


def run_video_inference(
    model_path: Path,
    video_path: Path,
    output_video: Path,
    output_json: Path,
    conf_threshold: float,
    input_size: int,
    compute_units,
):
    predictor = DFineCoreMLPredictor(
        str(model_path),
        conf_threshold=conf_threshold,
        input_size=input_size,
        compute_units=compute_units,
    )

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run D-FINE CoreML inference on a video and save outputs"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video",
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
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"CoreML model not found: {model_path}")

    output_video, output_json = get_output_paths(video_path, Path(args.output_dir))

    report = run_video_inference(
        model_path=model_path,
        video_path=video_path,
        output_video=output_video,
        output_json=output_json,
        conf_threshold=args.conf_threshold,
        input_size=args.input_size,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )

    print("\nInference complete.")
    print(f"JSON report  : {output_json}")
    print(f"Frames       : {report['frames_processed']}")
    print(f"Avg latency  : {report['average_latency_ms']:.2f} ms")
    print(f"Avg infer FPS: {report['average_inference_fps']:.2f}")


if __name__ == "__main__":
    main()