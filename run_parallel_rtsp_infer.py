import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


ALLOWED_COMPUTE_UNITS = {"auto", "all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run infer_coreml.py on the same input source in parallel workers."
    )
    parser.add_argument(
        "--input_source",
        default="rtsp://192.168.1.137:8554/live",
        help="Input source for each worker: RTSP URL, video file path, or directory.",
    )
    parser.add_argument(
        "--input_sources",
        default="",
        help=(
            "Comma-separated input sources for workers (overrides --input_source). "
            "If fewer than --num_streams, values are reused round-robin."
        ),
    )
    parser.add_argument(
        "--stream_url",
        default="",
        help="Deprecated alias for --input_source.",
    )
    parser.add_argument(
        "--num_streams",
        type=int,
        default=1,
        help="Number of parallel inference processes (e.g., 1, 2, 4).",
    )
    parser.add_argument(
        "--model",
        default="weights/dfine_x_obj2coco.mlpackage",
        help="Path to CoreML .mlpackage model.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/rtsp_parallel",
        help="Base output directory. One subfolder is created per stream.",
    )
    parser.add_argument(
        "--infer_script",
        default="infer_coreml.py",
        help="Path to infer_coreml.py.",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold forwarded to infer_coreml.py.",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=640,
        help="Input size forwarded to infer_coreml.py.",
    )
    parser.add_argument(
        "--compute_units",
        choices=["auto", "all", "cpu_and_gpu", "cpu_and_ne", "cpu_only"],
        default="auto",
        help="Compute units forwarded to infer_coreml.py.",
    )
    parser.add_argument(
        "--per_stream_compute_units",
        default="",
        help=(
            "Comma-separated compute units per worker, e.g. "
            "'cpu_and_gpu,cpu_and_ne'. If fewer than --num_streams, values are reused round-robin."
        ),
    )
    parser.add_argument(
        "--classes",
        default="all",
        help="Class filter forwarded to infer_coreml.py.",
    )
    parser.add_argument(
        "--start_delay",
        type=float,
        default=0.0,
        help="Delay in seconds between launching each process.",
    )
    parser.add_argument(
        "--stats_interval",
        type=float,
        default=2.0,
        help="Seconds between consolidated FPS summaries for all streams.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Run inference every Nth frame in each worker (1 = no skip).",
    )
    parser.add_argument(
        "--max_frames_mode",
        action="store_true",
        help="Disable rendering and video writing in workers for maximum throughput.",
    )
    parser.add_argument(
        "--no_max_frames_mode",
        action="store_false",
        dest="max_frames_mode",
        help="Keep rendering and output video writing in workers.",
    )
    parser.set_defaults(max_frames_mode=True)
    return parser.parse_args()


def is_stream_uri(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("rtsp://", "rtsps://", "http://", "https://"))


def _parse_csv_values(raw_value: str):
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _expand_to_count(values, count: int):
    return [values[idx % len(values)] for idx in range(count)]


def _validate_input_source(input_source: str):
    if is_stream_uri(input_source):
        return
    input_path = Path(input_source)
    if not input_path.exists():
        raise FileNotFoundError(f"input source not found: {input_path}")


def resolve_input_sources(args):
    if args.input_sources.strip():
        sources = _parse_csv_values(args.input_sources)
    else:
        source = args.stream_url.strip() or args.input_source.strip()
        sources = [source] if source else []

    if not sources:
        raise ValueError(
            "Please provide --input_source, --stream_url, or --input_sources."
        )

    for source in sources:
        _validate_input_source(source)

    return _expand_to_count(sources, args.num_streams)


def resolve_compute_units_per_stream(args):
    if args.per_stream_compute_units.strip():
        requested = _parse_csv_values(args.per_stream_compute_units)
    else:
        requested = [args.compute_units]

    if not requested:
        requested = [args.compute_units]

    for unit in requested:
        if unit not in ALLOWED_COMPUTE_UNITS:
            raise ValueError(
                f"Invalid compute unit '{unit}'. "
                f"Allowed: {sorted(ALLOWED_COMPUTE_UNITS)}"
            )

    return _expand_to_count(requested, args.num_streams)


def build_command(
    args,
    input_source: str,
    compute_units: str,
    stream_idx: int,
    stream_output_dir: Path,
):
    stream_name = f"stream_{stream_idx:02d}"
    cmd = [
        sys.executable,
        args.infer_script,
        "--input",
        input_source,
        "--model",
        args.model,
        "--output_dir",
        str(stream_output_dir),
        "--conf_threshold",
        str(args.conf_threshold),
        "--input_size",
        str(args.input_size),
        "--compute_units",
        compute_units,
        "--classes",
        args.classes,
        "--stream_label",
        stream_name,
        "--fps_log_interval",
        str(args.stats_interval),
        "--model_instance",
        f"model_{stream_idx:02d}",
        "--frame_stride",
        str(args.frame_stride),
    ]

    if args.max_frames_mode:
        cmd.extend(["--no_render", "--no_save_video"])

    return cmd


def _read_process_output(stream_name, proc, stats_by_stream, stats_lock):
    if proc.stdout is None:
        return

    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith("FPS_LOG "):
            payload = line[len("FPS_LOG ") :]
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                print(f"[{stream_name}] malformed FPS_LOG: {payload}")
                continue

            with stats_lock:
                stats_by_stream[stream_name] = {
                    "camera_fps": float(data.get("camera_fps", 0.0)),
                    "infer_fps": float(data.get("infer_fps", 0.0)),
                    "model_infer_fps": float(data.get("model_infer_fps", 0.0)),
                    "frames_total": int(data.get("frames_total", 0)),
                    "frames_inferred": int(data.get("frames_inferred", 0)),
                    "updated_at": time.time(),
                }
            continue

        print(f"[{stream_name}] {line}")


def _print_fps_snapshot(stats_by_stream, stats_lock, expected_streams: int):
    with stats_lock:
        snapshot = dict(stats_by_stream)

    if not snapshot:
        print("[FPS] waiting for stream metrics...")
        return None

    stream_names = sorted(snapshot.keys())
    per_stream_chunks = []
    total_camera_fps = 0.0
    total_infer_fps = 0.0
    total_model_infer_fps = 0.0

    for stream_name in stream_names:
        metrics = snapshot[stream_name]
        camera_fps = float(metrics.get("camera_fps", 0.0))
        infer_fps = float(metrics.get("infer_fps", 0.0))
        model_infer_fps = float(metrics.get("model_infer_fps", 0.0))
        frames_total = int(metrics.get("frames_total", 0))
        frames_inferred = int(metrics.get("frames_inferred", 0))

        total_camera_fps += camera_fps
        total_infer_fps += infer_fps
        total_model_infer_fps += model_infer_fps
        per_stream_chunks.append(
            f"{stream_name}: cam={camera_fps:.2f} infer={infer_fps:.2f} model={model_infer_fps:.2f} inferred={frames_inferred} read={frames_total}"
        )

    active_streams = len(stream_names)
    avg_camera_fps = total_camera_fps / active_streams if active_streams else 0.0
    avg_infer_fps = total_infer_fps / active_streams if active_streams else 0.0
    avg_model_infer_fps = total_model_infer_fps / active_streams if active_streams else 0.0

    print("[FPS] " + " | ".join(per_stream_chunks))
    print(
        "[FPS][ALL] "
        f"active={active_streams}/{expected_streams} "
        f"sum_cam={total_camera_fps:.2f} avg_cam={avg_camera_fps:.2f} "
        f"sum_infer={total_infer_fps:.2f} avg_infer={avg_infer_fps:.2f} "
        f"sum_model={total_model_infer_fps:.2f} avg_model={avg_model_infer_fps:.2f}"
    )

    return {
        "timestamp": time.perf_counter(),
        "active_streams": active_streams,
        "sum_infer_fps": total_infer_fps,
        "sum_model_infer_fps": total_model_infer_fps,
    }


def _summarize_parallel_window_fps(fps_snapshots):
    if not fps_snapshots:
        return {
            "peak_sum_infer_fps": 0.0,
            "peak_sum_model_fps": 0.0,
            "avg_sum_infer_fps": 0.0,
            "avg_sum_model_fps": 0.0,
            "active_window_s": 0.0,
        }

    peak_sum_infer_fps = max(item["sum_infer_fps"] for item in fps_snapshots)
    peak_sum_model_fps = max(item["sum_model_infer_fps"] for item in fps_snapshots)

    weighted_sum_infer = 0.0
    weighted_sum_model = 0.0
    weighted_duration_s = 0.0

    for previous, current in zip(fps_snapshots, fps_snapshots[1:]):
        delta_s = max(current["timestamp"] - previous["timestamp"], 0.0)
        if delta_s <= 0:
            continue
        weighted_sum_infer += previous["sum_infer_fps"] * delta_s
        weighted_sum_model += previous["sum_model_infer_fps"] * delta_s
        weighted_duration_s += delta_s

    if weighted_duration_s > 0:
        avg_sum_infer_fps = weighted_sum_infer / weighted_duration_s
        avg_sum_model_fps = weighted_sum_model / weighted_duration_s
    else:
        avg_sum_infer_fps = sum(item["sum_infer_fps"] for item in fps_snapshots) / len(
            fps_snapshots
        )
        avg_sum_model_fps = sum(
            item["sum_model_infer_fps"] for item in fps_snapshots
        ) / len(fps_snapshots)

    return {
        "peak_sum_infer_fps": peak_sum_infer_fps,
        "peak_sum_model_fps": peak_sum_model_fps,
        "avg_sum_infer_fps": avg_sum_infer_fps,
        "avg_sum_model_fps": avg_sum_model_fps,
        "active_window_s": weighted_duration_s,
    }


def _print_final_aggregate_inference_summary(
    processes,
    run_wall_time_s: float,
    run_start_epoch_s: float,
    fps_snapshots,
):
    total_frames_read = 0
    total_frames_inferred = 0
    total_frames_skipped = 0
    total_video_elapsed_s = 0.0
    total_model_infer_elapsed_s = 0.0
    parsed_reports = 0
    missing_reports = []

    for _, stream_name, _, stream_output_dir in processes:
        all_report_paths = sorted(
            stream_output_dir.glob("*_coreml_report*.json"),
            key=lambda path: path.stat().st_mtime,
        )
        report_paths = [
            path for path in all_report_paths if path.stat().st_mtime >= run_start_epoch_s - 1.0
        ]

        if not report_paths:
            missing_reports.append(stream_name)
            continue

        for report_path in report_paths:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[{stream_name}] failed to read report {report_path}: {exc}")
                continue

            parsed_reports += 1

            frames_read = int(report.get("frames_read", 0))
            frames_inferred = int(report.get("frames_inferred", 0))
            frames_skipped = int(
                report.get("frames_skipped", max(frames_read - frames_inferred, 0))
            )
            effective_infer_fps = float(report.get("effective_inference_fps", 0.0))
            avg_latency_ms = float(report.get("average_latency_ms", 0.0))

            total_frames_read += frames_read
            total_frames_inferred += frames_inferred
            total_frames_skipped += frames_skipped

            if effective_infer_fps > 0:
                total_video_elapsed_s += frames_inferred / effective_infer_fps

            if avg_latency_ms > 0:
                total_model_infer_elapsed_s += (frames_inferred * avg_latency_ms) / 1000.0

    if parsed_reports == 0:
        print("\nAggregate inference summary: unavailable (no JSON reports found).")
        return

    frame_weighted_effective_fps = (
        total_frames_inferred / total_video_elapsed_s if total_video_elapsed_s > 0 else 0.0
    )
    frame_weighted_model_fps = (
        total_frames_inferred / total_model_infer_elapsed_s
        if total_model_infer_elapsed_s > 0
        else 0.0
    )
    wall_clock_throughput_fps = (
        total_frames_inferred / run_wall_time_s if run_wall_time_s > 0 else 0.0
    )
    window_summary = _summarize_parallel_window_fps(fps_snapshots)

    print("\nAggregate inference summary")
    print(f"Reports parsed: {parsed_reports}")
    print(f"Frames read (all streams): {total_frames_read}")
    print(f"Frames inferred (all streams): {total_frames_inferred}")
    print(f"Frames skipped (all streams): {total_frames_skipped}")
    print(
        "Average effective inference FPS (frame-weighted): "
        f"{frame_weighted_effective_fps:.2f}"
    )
    print(
        "Average model-only inference FPS (latency-weighted): "
        f"{frame_weighted_model_fps:.2f}"
    )
    print(
        "Overall throughput FPS (wall-clock, stagger-aware): "
        f"{wall_clock_throughput_fps:.2f} over {run_wall_time_s:.2f}s"
    )
    print(
        "Peak parallel inferred FPS (window sum): "
        f"{window_summary['peak_sum_infer_fps']:.2f}"
    )
    print(
        "Average parallel inferred FPS (active window): "
        f"{window_summary['avg_sum_infer_fps']:.2f}"
    )
    print(
        "Peak parallel model FPS (window sum): "
        f"{window_summary['peak_sum_model_fps']:.2f}"
    )
    print(
        "Average parallel model FPS (active window): "
        f"{window_summary['avg_sum_model_fps']:.2f}"
    )
    if missing_reports:
        print(f"Missing stream reports: {', '.join(sorted(missing_reports))}")


def main():
    args = parse_args()

    if args.num_streams < 1:
        raise ValueError("--num_streams must be >= 1")

    infer_script = Path(args.infer_script)
    model_path = Path(args.model)
    output_base = Path(args.output_dir)

    input_sources_per_stream = resolve_input_sources(args)
    compute_units_per_stream = resolve_compute_units_per_stream(args)

    if not infer_script.exists():
        raise FileNotFoundError(f"infer script not found: {infer_script}")
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    output_base.mkdir(parents=True, exist_ok=True)

    stats_by_stream = {}
    stats_lock = threading.Lock()
    processes = []
    reader_threads = []
    fps_snapshots = []

    print(f"Launching {args.num_streams} parallel stream(s)")
    print(f"Input source[1]: {input_sources_per_stream[0]}")
    if len(set(input_sources_per_stream)) > 1:
        print("Input assignment: per-stream sources enabled")
    else:
        print("Input assignment: same source reused for all streams")
    print(f"Model: {model_path}")
    print("Dedicated model instances: enabled (one per stream process)")
    print(f"Max frames mode: {'ON' if args.max_frames_mode else 'OFF'}")
    print(f"Frame stride: {args.frame_stride}")

    parallel_run_start = time.perf_counter()
    parallel_run_start_epoch = time.time()

    for stream_idx in range(1, args.num_streams + 1):
        input_source = input_sources_per_stream[stream_idx - 1]
        stream_compute_units = compute_units_per_stream[stream_idx - 1]
        stream_name = f"stream_{stream_idx:02d}"
        stream_output_dir = output_base / f"stream_{stream_idx:02d}"
        stream_output_dir.mkdir(parents=True, exist_ok=True)
        cmd = build_command(
            args,
            input_source,
            stream_compute_units,
            stream_idx,
            stream_output_dir,
        )

        print(
            f"[{stream_idx}] source={input_source} compute_units={stream_compute_units}"
        )
        print(f"[{stream_idx}] Starting: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append((stream_idx, stream_name, proc, stream_output_dir))

        reader = threading.Thread(
            target=_read_process_output,
            args=(stream_name, proc, stats_by_stream, stats_lock),
            daemon=True,
        )
        reader.start()
        reader_threads.append(reader)

        if args.start_delay > 0 and stream_idx < args.num_streams:
            time.sleep(args.start_delay)

    exit_codes = []
    next_fps_log_time = time.perf_counter() + max(args.stats_interval, 0.1)

    try:
        while True:
            now = time.perf_counter()
            if now >= next_fps_log_time:
                fps_snapshot = _print_fps_snapshot(
                    stats_by_stream, stats_lock, args.num_streams
                )
                if fps_snapshot is not None:
                    fps_snapshots.append(fps_snapshot)
                next_fps_log_time = now + max(args.stats_interval, 0.1)

            if all(proc.poll() is not None for _, _, proc, _ in processes):
                break

            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Stopping child processes...")
        for _, _, proc, _ in processes:
            if proc.poll() is None:
                proc.terminate()

        for _, _, proc, _ in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise

    for stream_idx, stream_name, proc, stream_output_dir in processes:
        code = proc.poll()
        if code is None:
            code = proc.wait()
        exit_codes.append(code)
        print(f"[{stream_idx}] Exit code: {code} (output: {stream_output_dir})")

    for reader in reader_threads:
        reader.join(timeout=1)

    final_snapshot = _print_fps_snapshot(stats_by_stream, stats_lock, args.num_streams)
    if final_snapshot is not None:
        fps_snapshots.append(final_snapshot)
    run_wall_time_s = max(time.perf_counter() - parallel_run_start, 1e-9)
    _print_final_aggregate_inference_summary(
        processes,
        run_wall_time_s,
        parallel_run_start_epoch,
        fps_snapshots,
    )

    failed = sum(1 for code in exit_codes if code != 0)
    succeeded = len(exit_codes) - failed
    print("\nSummary")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
