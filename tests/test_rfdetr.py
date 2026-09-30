from pathlib import Path

import numpy as np

from frame_segmentation.models import RFDetrSegmenter


class _Detections:
    xyxy = np.asarray([[3, 4, 13, 14]])
    class_id = np.asarray([2])
    confidence = np.asarray([0.87])
    mask = np.asarray([[[1, 0], [0, 1]]])
    metadata = {"source_image": np.zeros((24, 32, 3), dtype=np.uint8)}


class _FakeRFDetr:
    def __init__(self):
        self.calls = []

    def predict(self, image, threshold):
        self.calls.append((image, threshold))
        return _Detections()


def test_rfdetr_segmenter_normalizes_detections():
    fake_model = _FakeRFDetr()
    segmenter = RFDetrSegmenter(
        model_size="small",
        model_factory=lambda _: fake_model,
        class_names={2: "car"},
    )

    prediction = segmenter.predict(Path("frame.jpg"), confidence=0.4)

    assert prediction.image_size == (32, 24)
    assert len(prediction.detections) == 1
    detection = prediction.detections[0]
    assert detection.label == "car"
    assert detection.class_id == 2
    assert detection.confidence == 0.87
    assert detection.box_xyxy == (3.0, 4.0, 13.0, 14.0)
    assert detection.mask.tolist() == [[True, False], [False, True]]
    assert fake_model.calls == [("frame.jpg", 0.4)]
