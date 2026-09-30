from frame_segmentation.cli import main
from frame_segmentation.runner import RunArtifacts


def test_cli_constructs_yolo_backend_and_runs_inference(tmp_path, capsys):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"placeholder")
    created = []
    calls = []

    def factory(backend, **kwargs):
        created.append((backend, kwargs))
        return object()

    def runner(segmenter, input_path, output_dir, confidence):
        calls.append((segmenter, input_path, output_dir, confidence))
        output_dir.mkdir(parents=True)
        overlay = output_dir / "overlay.png"
        predictions = output_dir / "predictions.json"
        overlay.touch()
        predictions.touch()
        return RunArtifacts(output_dir, overlay, predictions)

    exit_code = main(
        [
            "--backend",
            "yolo",
            "--model",
            "yolo11m-seg.pt",
            "--input",
            str(image),
            "--output-dir",
            str(tmp_path / "out"),
            "--confidence",
            "0.4",
            "--device",
            "cuda:0",
        ],
        segmenter_factory=factory,
        inference_runner=runner,
    )

    assert exit_code == 0
    assert created == [
        ("yolo", {"model_name": "yolo11m-seg.pt", "device": "cuda:0"})
    ]
    assert calls[0][1:] == (image, tmp_path / "out", 0.4)
    assert "overlay.png" in capsys.readouterr().out


def test_cli_passes_custom_checkpoint_to_safely_configured_yolo_segmenter(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"placeholder")
    created = []

    def factory(backend, **kwargs):
        created.append((backend, kwargs))
        return object()

    def runner(segmenter, input_path, output_dir, confidence):
        return RunArtifacts(output_dir, output_dir / "overlay.png", output_dir / "predictions.json")

    exit_code = main(
        [
            "--backend",
            "yolo",
            "--model",
            str(tmp_path / "custom.pt"),
            "--input",
            str(image),
        ],
        segmenter_factory=factory,
        inference_runner=runner,
    )

    assert exit_code == 0
    assert created == [
        (
            "yolo",
            {
                "model_name": str(tmp_path / "custom.pt"),
                "device": "cuda",
            },
        )
    ]
