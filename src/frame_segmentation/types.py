from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None


@dataclass(slots=True)
class Prediction:
    image_size: tuple[int, int]
    detections: list[Detection]
