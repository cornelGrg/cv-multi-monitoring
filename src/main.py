"""
Multi-Camera Real-Time Traffic Analytics Pipeline — entry point.

Usage::

    python -m src.main                              # uses configs/default.yaml
    python -m src.main --config configs/benchmark_replicated_streams.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.inference.engine import OnnxGpuEngine
from src.metrics.collector import MetricsCollector
from src.pipeline.orchestrator import PipelineOrchestrator


def load_config(path: str | Path) -> dict:
    """Load and return a YAML configuration file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open() as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def main(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
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

    logger.info("Model input:  %s → %s", engine.input_name, engine.input_shape)
    logger.info("Model output: %s", engine.output_shape)

    # ── Metrics collector ───────────────────────────────────────────────
    camera_ids = [s["camera_id"] for s in config["streams"]]
    collector = MetricsCollector(camera_ids)

    # ── Orchestrator ────────────────────────────────────────────────────
    orchestrator = PipelineOrchestrator(config, engine, collector)
    orchestrator.run()

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
    args = parser.parse_args()
    main(args.config)
