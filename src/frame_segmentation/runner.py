from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .types import Prediction


class Segmenter(Protocol):
    def predict(self, image_path: Path, confidence: float) -> Prediction: ...


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    output_dir: Path
    overlay_path: Path
    json_path: Path
    inference_ms: float = 0.0


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return cleaned or "object"


def _fit_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    binary = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255)
    if binary.size != size:
        binary = binary.resize(size, resample=Image.Resampling.NEAREST)
    return np.asarray(binary) > 0


def run_inference(
    segmenter: Segmenter,
    input_path: Path,
    output_dir: Path,
    confidence: float,
    clock: Callable[[], float] = perf_counter,
) -> RunArtifacts:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True)

    started_at = clock()
    prediction = segmenter.predict(input_path, confidence)
    inference_ms = round((clock() - started_at) * 1000, 3)
    image = Image.open(input_path).convert("RGB")
    if image.size != prediction.image_size:
        raise ValueError(
            f"Prediction size {prediction.image_size} does not match image size {image.size}"
        )

    overlay = np.asarray(image).copy()
    palette = [(239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8)]
    records = []

    for index, detection in enumerate(prediction.detections):
        color = np.asarray(palette[index % len(palette)], dtype=np.float32)
        mask_file = None
        if detection.mask is not None:
            mask = _fit_mask(detection.mask, image.size)
            overlay[mask] = (overlay[mask] * 0.55 + color * 0.45).astype(np.uint8)
            mask_file = f"masks/{index:03d}-{_safe_name(detection.label)}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(output_dir / mask_file)

        records.append(
            {
                "class_id": detection.class_id,
                "label": detection.label,
                "confidence": detection.confidence,
                "box_xyxy": list(detection.box_xyxy),
                "mask_file": mask_file,
            }
        )

    annotated = Image.fromarray(overlay)
    draw = ImageDraw.Draw(annotated)
    for index, detection in enumerate(prediction.detections):
        color = palette[index % len(palette)]
        draw.rectangle(detection.box_xyxy, outline=color, width=2)
        draw.text(
            (detection.box_xyxy[0] + 2, detection.box_xyxy[1] + 2),
            f"{detection.label} {detection.confidence:.2f}",
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )

    overlay_path = output_dir / "overlay.png"
    json_path = output_dir / "predictions.json"
    annotated.save(overlay_path)
    json_path.write_text(
        json.dumps(
            {
                "image": str(input_path),
                "width": prediction.image_size[0],
                "height": prediction.image_size[1],
                "inference_ms": inference_ms,
                "detections": records,
            },
            indent=2,
        )
        + "\n"
    )
    return RunArtifacts(output_dir, overlay_path, json_path, inference_ms)
