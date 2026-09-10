"""
Multi-Camera Real-Time Traffic Analytics Pipeline — entry point.

Usage::

    python -m src.main                              # uses configs/default.yaml
    python -m src.main --config configs/benchmark_realtime_detection_compute.yaml
"""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import yaml

from src.inference.engine import OnnxGpuEngine
from src.metrics.collector import MetricsCollector
from src.pipeline.orchestrator import PipelineOrchestrator
from src.tracking.bytetrack import VEHICLE_CLASSES


def require_active_provider(engine: OnnxGpuEngine, inference_config: dict) -> None:
    """Fail before benchmarking when the requested execution provider fell back."""
    if not inference_config.get("require_provider", False):
        return

    requested = inference_config["provider"]
    active = engine.providers
    if not active or active[0] != requested:
        raise RuntimeError(
            f"Required provider {requested!r} is not active; session providers: {active}. "
            "Benchmark aborted to prevent recording CPU fallback results."
        )


def load_config(path: str | Path) -> dict:
    """Load and return a YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        config = yaml.safe_load(f)

    if "manifest_file" in config:
        manifest_path = p.parent.parent / config["manifest_file"]
        with manifest_path.open() as mf:
            manifest_data = yaml.safe_load(mf)
        config["streams"] = manifest_data.get("cameras", [])
        for stream in config["streams"]:
            if "path" not in stream:
                stream["path"] = stream["video_path"]
        if "source_fps" in manifest_data.get("dataset", {}):
            config.setdefault("pipeline", {})["source_fps"] = manifest_data["dataset"]["source_fps"]

    return config


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main(config_path: str = "configs/default.yaml", *, dashboard_port: int | None = None) -> None:
    config = load_config(config_path)
    if dashboard_port is not None:
        from src.output.dashboard import dashboard_config

        config = dashboard_config(config)
    setup_logging(config.get("output", {}).get("log_level", "INFO"))

    logger = logging.getLogger(__name__)
    logger.info("Loaded config: %s", config_path)

    project_root = Path(__file__).resolve().parent.parent

    # ── Inference engine ────────────────────────────────────────────────
    model_path = project_root / config["model"]["path"]
    inf_cfg = config["inference"]

    engine = OnnxGpuEngine(
        model_path=model_path,
        provider=inf_cfg["provider"],
        device_id=inf_cfg["device_id"],
        warmup_batches=config["pipeline"]["warmup_batches"],
    )
    require_active_provider(engine, inf_cfg)

    logger.info("Model input:  %s → %s", engine.input_name, engine.input_shape)
    logger.info("Model output: %s", engine.output_shape)
    logger.info("Active execution providers: %s", engine.providers)

    # ── Metrics collector ───────────────────────────────────────────────
    camera_ids = [s["camera_id"] for s in config["streams"]]
    pipeline_cfg = config.get("pipeline", {})
    tracking_cfg = config.get("tracking", {})
    analytics_cfg = config.get("analytics", {})
    output_cfg = config.get("output", {})
    allowed_class_ids = pipeline_cfg.get("allowed_classes", [2, 3, 5, 7])
    try:
        recorded_config_path = str(Path(config_path).resolve().relative_to(project_root))
    except ValueError:
        recorded_config_path = str(config_path)
    run_metadata = {
        "config": recorded_config_path,
        "mode": (
            "realtime_simulation"
            if pipeline_cfg.get("source_fps")
            else "max_throughput"
        ),
        "model": config["model"]["path"],
        "model_precision": "FP16",
        "input_size": config["model"]["input_size"],
        "confidence_threshold": config["model"]["confidence_threshold"],
        "postprocessing": (
            "native end-to-end model output without a separate NMS pass; configured iou_threshold "
            "is informational and is not applied again in Python"
        ),
        "iou_threshold_configured": config["model"].get("iou_threshold"),
        "allowed_class_ids": allowed_class_ids,
        "allowed_class_names": [
            VEHICLE_CLASSES.get(class_id, str(class_id))
            for class_id in allowed_class_ids
        ],
        "requested_provider": inf_cfg["provider"],
        "active_providers": engine.providers,
        "batch_size": inf_cfg["batch_size"],
        "num_streams": len(camera_ids),
        "source_fps": pipeline_cfg.get("source_fps"),
        "duration_s": pipeline_cfg.get("max_duration_s", 0.0),
        "tracking_enabled": bool(tracking_cfg.get("enabled", False)),
        "show_detections": bool(output_cfg.get("show_detections", False)),
        "analytics_enabled": bool(analytics_cfg.get("enabled", False)),
        "traffic_flow_enabled": bool(
            analytics_cfg.get("enabled", False)
            and analytics_cfg.get("flow", {}).get("enabled", False)
        ),
        "traffic_flow_window_s": analytics_cfg.get("flow", {}).get("window_s", 60.0),
        "analytics_coordinate_space": (
            "normalized_letterbox_canvas"
            if analytics_cfg.get("enabled", False)
            else None
        ),
        "analytics_lines_by_camera": {
            camera_id: [line.get("line_id") for line in camera_cfg.get("lines", [])]
            for camera_id, camera_cfg in analytics_cfg.get("cameras", {}).items()
        },
        "video_encoder": output_cfg.get(
            "video_encoder",
            tracking_cfg.get("video_encoder"),
        ),
    }
    logger.info(
        "Detection filter: confidence > %.2f, allowed COCO classes: %s",
        config["model"]["confidence_threshold"],
        dict(zip(allowed_class_ids, run_metadata["allowed_class_names"])),
    )
    collector = MetricsCollector(
        camera_ids, run_metadata=run_metadata,
        sample_limit=2000 if dashboard_port is not None else None,
    )

    # ── Orchestrator ────────────────────────────────────────────────────
    dashboard_context = nullcontext()
    if dashboard_port is not None:
        from src.output.dashboard import DashboardServer

        dashboard_context = DashboardServer(config, port=dashboard_port)
    with dashboard_context as dashboard:
        orchestrator = PipelineOrchestrator(
            config, engine, collector,
            frame_sink=dashboard.publish if dashboard is not None else None,
        )
        try:
            orchestrator.run()
        except KeyboardInterrupt:
            logger.info("Stopped by user")

    # ── Results ─────────────────────────────────────────────────────────
    collector.print_summary()

    metrics_path = project_root / config["output"]["metrics_file"]
    collector.save(metrics_path)

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-Camera Traffic Analytics Pipeline"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument("--dashboard", action="store_true", help="Serve a live browser demo of looping clips")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard localhost port (default: 8000)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    main(args.config, dashboard_port=args.port if args.dashboard else None)
