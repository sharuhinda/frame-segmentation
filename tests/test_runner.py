import json
from pathlib import Path

import numpy as np
from PIL import Image

from frame_segmentation.runner import run_inference
from frame_segmentation.types import Detection, Prediction


class _FakeSegmenter:
    def predict(self, image_path: Path, confidence: float) -> Prediction:
        assert image_path.name == "frame.png"
        assert confidence == 0.3
        return Prediction(
            image_size=(8, 6),
            detections=[
                Detection(
                    class_id=1,
                    label="bicycle",
                    confidence=0.95,
                    box_xyxy=(1.0, 1.0, 6.0, 5.0),
                    mask=np.asarray(
                        [
                            [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=bool,
                    ),
                )
            ],
        )


def test_run_inference_writes_overlay_and_machine_readable_json(tmp_path):
    input_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), "white").save(input_path)

    artifacts = run_inference(
        segmenter=_FakeSegmenter(),
        input_path=input_path,
        output_dir=tmp_path / "output",
        confidence=0.3,
        clock=iter([1.0, 1.125]).__next__,
    )

    assert artifacts.overlay_path.exists()
    assert Image.open(artifacts.overlay_path).size == (8, 6)
    payload = json.loads(artifacts.json_path.read_text())
    assert payload == {
        "image": str(input_path),
        "width": 8,
        "height": 6,
        "inference_ms": 125.0,
        "detections": [
            {
                "class_id": 1,
                "label": "bicycle",
                "confidence": 0.95,
                "box_xyxy": [1.0, 1.0, 6.0, 5.0],
                "mask_file": "masks/000-bicycle.png",
            }
        ],
    }
    assert (artifacts.output_dir / "masks/000-bicycle.png").exists()


def test_run_inference_clears_only_stale_masks_on_rerun(tmp_path):
    input_path = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), "white").save(input_path)
    output_dir = tmp_path / "output"
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True)
    stale_mask = masks_dir / "stale.png"
    stale_mask.write_bytes(b"stale")
    unrelated_output = output_dir / "keep.txt"
    unrelated_output.write_text("keep")

    run_inference(
        segmenter=_FakeSegmenter(),
        input_path=input_path,
        output_dir=output_dir,
        confidence=0.3,
    )

    assert not stale_mask.exists()
    assert (masks_dir / "000-bicycle.png").exists()
    assert unrelated_output.read_text() == "keep"
