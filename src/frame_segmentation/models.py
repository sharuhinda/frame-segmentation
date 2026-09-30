from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .types import Detection, Prediction


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloSegmenter:
    def __init__(
        self,
        model_name: str = "yolo11s-seg.pt",
        device: str = "cuda",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        os.environ["ULTRALYTICS_SAFE_LOAD"] = "1"
        if model_factory is None:
            try:
                from ultralytics import YOLO
                from ultralytics.nn import tasks as ultralytics_tasks
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO support is not installed. Run: uv sync --extra yolo"
                ) from exc
            if not hasattr(ultralytics_tasks, "SAFE_LOAD"):
                raise RuntimeError(
                    "This Ultralytics version does not support restricted checkpoint loading. "
                    "Run: uv sync --upgrade --extra yolo"
                )
            safe_loader = getattr(ultralytics_tasks, "_SafeLoad", None)
            if safe_loader is None or not safe_loader.SUPPORTED:
                raise RuntimeError(
                    "Restricted Ultralytics checkpoint loading requires torch>=2.6; "
                    "upgrade PyTorch before loading YOLO checkpoints."
                )
            ultralytics_tasks.SAFE_LOAD = True
            model_factory = YOLO
        self.model = model_factory(model_name)
        self.device = device

    def predict(self, image_path: Path, confidence: float = 0.25) -> Prediction:
        result = self.model.predict(
            source=str(image_path),
            conf=confidence,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )[0]

        boxes = _as_numpy(result.boxes.xyxy)
        class_ids = _as_numpy(result.boxes.cls).astype(int)
        scores = _as_numpy(result.boxes.conf)
        masks = None if result.masks is None else _as_numpy(result.masks.data).astype(bool)

        detections = []
        for index, (box, class_id, score) in enumerate(
            zip(boxes, class_ids, scores, strict=True)
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    label=str(result.names[int(class_id)]),
                    confidence=float(score),
                    box_xyxy=tuple(float(value) for value in box),
                    mask=None if masks is None else masks[index],
                )
            )

        height, width = result.orig_shape
        return Prediction(image_size=(int(width), int(height)), detections=detections)


class RFDetrSegmenter:
    def __init__(
        self,
        model_size: str = "small",
        model_factory: Callable[[str], Any] | None = None,
        class_names: dict[int, str] | list[str] | None = None,
    ) -> None:
        if model_factory is None:
            try:
                import rfdetr
                from rfdetr.assets.coco_classes import COCO_CLASSES
            except ImportError as exc:
                raise RuntimeError(
                    "RF-DETR support is not installed. Run: uv sync --extra rfdetr"
                ) from exc

            model_classes = {
                "nano": rfdetr.RFDETRSegNano,
                "small": rfdetr.RFDETRSegSmall,
                "medium": rfdetr.RFDETRSegMedium,
                "large": rfdetr.RFDETRSegLarge,
            }
            try:
                model_class = model_classes[model_size]
            except KeyError as exc:
                choices = ", ".join(model_classes)
                raise ValueError(
                    f"Unknown RF-DETR model size {model_size!r}; choose from {choices}"
                ) from exc
            def model_factory(_: str) -> Any:
                return model_class()

            class_names = COCO_CLASSES

        self.model = model_factory(model_size)
        self.class_names = class_names or {}

    def _label_for(self, class_id: int) -> str:
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(class_id, class_id))
        if 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return str(class_id)

    def predict(self, image_path: Path, confidence: float = 0.25) -> Prediction:
        result = self.model.predict(str(image_path), threshold=confidence)
        boxes = np.asarray(result.xyxy)
        class_ids = np.asarray(result.class_id).astype(int)
        scores = np.asarray(result.confidence)
        raw_masks = getattr(result, "mask", None)
        masks = None if raw_masks is None else np.asarray(raw_masks).astype(bool)

        detections = []
        for index, (box, class_id, score) in enumerate(
            zip(boxes, class_ids, scores, strict=True)
        ):
            detections.append(
                Detection(
                    class_id=int(class_id),
                    label=self._label_for(int(class_id)),
                    confidence=float(score),
                    box_xyxy=tuple(float(value) for value in box),
                    mask=None if masks is None else masks[index],
                )
            )

        source_image = result.metadata["source_image"]
        height, width = source_image.shape[:2]
        return Prediction(image_size=(int(width), int(height)), detections=detections)


def create_segmenter(backend: str, **kwargs: Any) -> YoloSegmenter | RFDetrSegmenter:
    if backend == "yolo":
        return YoloSegmenter(**kwargs)
    if backend == "rfdetr":
        return RFDetrSegmenter(**kwargs)
    raise ValueError(f"Unknown backend: {backend}")
