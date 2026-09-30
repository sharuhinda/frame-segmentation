import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from frame_segmentation.models import YoloSegmenter


class _Tensor:
    def __init__(self, value):
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self._value


class _Boxes:
    xyxy = _Tensor([[1, 2, 11, 12]])
    cls = _Tensor([0])
    conf = _Tensor([0.91])


class _Masks:
    data = _Tensor([[[0, 1], [1, 0]]])


class _Result:
    names = {0: "person"}
    boxes = _Boxes()
    masks = _Masks()
    orig_shape = (20, 30)


class _FakeYolo:
    def __init__(self):
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_Result()]


def test_yolo_enables_restricted_loading_before_constructing_model(monkeypatch):
    fake_ultralytics = ModuleType("ultralytics")
    fake_nn = ModuleType("ultralytics.nn")
    fake_tasks = ModuleType("ultralytics.nn.tasks")
    fake_tasks.SAFE_LOAD = False
    fake_tasks._SafeLoad = type("_SafeLoad", (), {"SUPPORTED": True})
    observed = []

    class FakeYolo:
        def __init__(self, model_name):
            observed.append(
                (model_name, fake_tasks.SAFE_LOAD, os.environ.get("ULTRALYTICS_SAFE_LOAD"))
            )

    fake_ultralytics.YOLO = FakeYolo
    fake_nn.tasks = fake_tasks
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.tasks", fake_tasks)
    monkeypatch.delenv("ULTRALYTICS_SAFE_LOAD", raising=False)

    YoloSegmenter(model_name="checkpoints/custom.pt")

    assert observed == [("checkpoints/custom.pt", True, "1")]


def test_yolo_refuses_runtime_without_restricted_loader_support(monkeypatch):
    fake_ultralytics = ModuleType("ultralytics")
    fake_nn = ModuleType("ultralytics.nn")
    fake_tasks = ModuleType("ultralytics.nn.tasks")
    fake_tasks.SAFE_LOAD = False
    fake_tasks._SafeLoad = type("_SafeLoad", (), {"SUPPORTED": False})
    constructed = []

    class FakeYolo:
        def __init__(self, model_name):
            constructed.append(model_name)

    fake_ultralytics.YOLO = FakeYolo
    fake_nn.tasks = fake_tasks
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setitem(sys.modules, "ultralytics.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "ultralytics.nn.tasks", fake_tasks)

    with pytest.raises(RuntimeError, match="torch>=2.6"):
        YoloSegmenter(model_name="yolo11s-seg.pt")

    assert constructed == []


def test_yolo_segmenter_normalizes_boxes_labels_scores_and_masks():
    fake_model = _FakeYolo()
    segmenter = YoloSegmenter(
        model_name="yolo11n-seg.pt",
        device="cuda",
        model_factory=lambda _: fake_model,
    )

    prediction = segmenter.predict(Path("frame.jpg"), confidence=0.25)

    assert prediction.image_size == (30, 20)
    assert len(prediction.detections) == 1
    detection = prediction.detections[0]
    assert detection.label == "person"
    assert detection.class_id == 0
    assert detection.confidence == 0.91
    assert detection.box_xyxy == (1.0, 2.0, 11.0, 12.0)
    assert detection.mask.tolist() == [[False, True], [True, False]]
    assert fake_model.calls == [
        {
            "source": "frame.jpg",
            "conf": 0.25,
            "device": "cuda",
            "retina_masks": True,
            "verbose": False,
        }
    ]
