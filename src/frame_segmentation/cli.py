from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .models import create_segmenter
from .runner import RunArtifacts, run_inference


def _confidence(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO or RF-DETR instance segmentation on one image."
    )
    parser.add_argument("--backend", choices=("yolo", "rfdetr"), required=True)
    parser.add_argument(
        "--model",
        help="YOLO checkpoint name/path, or RF-DETR size: nano, small, medium, large",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/latest"))
    parser.add_argument("--confidence", type=_confidence, default=0.25)
    parser.add_argument("--device", default="cuda", help="YOLO device, e.g. cuda, cuda:0, cpu")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    segmenter_factory: Callable[..., Any] = create_segmenter,
    inference_runner: Callable[..., RunArtifacts] = run_inference,
) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"Input image does not exist: {args.input}")

    if args.backend == "yolo":
        segmenter = segmenter_factory(
            "yolo",
            model_name=args.model or "yolo11s-seg.pt",
            device=args.device,
        )
    else:
        segmenter = segmenter_factory(
            "rfdetr",
            model_size=args.model or "small",
        )

    artifacts = inference_runner(
        segmenter=segmenter,
        input_path=args.input,
        output_dir=args.output_dir,
        confidence=args.confidence,
    )
    print(f"Inference: {artifacts.inference_ms:.3f} ms")
    print(f"Overlay: {artifacts.overlay_path}")
    print(f"Predictions: {artifacts.json_path}")
    return 0
