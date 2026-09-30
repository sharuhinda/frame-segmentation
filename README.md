# frame-segmentation

A minimal local harness for comparing **YOLO11 segmentation** and **RF-DETR segmentation** on the same image. Both adapters produce the same JSON schema, individual binary masks, and an annotated overlay.

## Requirements

- Python 3.10+
- PyTorch 2.6+ for restricted YOLO checkpoint loading
- An NVIDIA CUDA-capable PyTorch installation is recommended
- Model checkpoints are downloaded automatically on first use

The default models are deliberately modest enough for a 12 GB GPU:

- YOLO: `yolo11s-seg.pt`
- RF-DETR: `small` (`RFDETRSegSmall`)

## Install

[`uv`](https://docs.astral.sh/uv/) is the recommended environment manager.

```bash
git clone https://github.com/sharuhinda/frame-segmentation.git
cd frame-segmentation

# Install both model backends.
uv sync --extra models
```

Install just one backend if preferred:

```bash
uv sync --extra yolo
# or
uv sync --extra rfdetr
```

Check that PyTorch can see the GPU:

```bash
uv run python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else False)"
```

If this prints `False`, install the PyTorch build appropriate for your NVIDIA driver and CUDA environment before running inference.

## Try YOLO

```bash
uv run frame-segment \
  --backend yolo \
  --model yolo11s-seg.pt \
  --input path/to/frame.jpg \
  --output-dir runs/yolo
```

Try another official checkpoint or CPU execution:

```bash
uv run frame-segment --backend yolo --model yolo11m-seg.pt \
  --device cuda:0 --input path/to/frame.jpg --output-dir runs/yolo-m

uv run frame-segment --backend yolo --device cpu \
  --input path/to/frame.jpg --output-dir runs/yolo-cpu
```

YOLO restricted checkpoint loading is forced on: the adapter uses PyTorch
`weights_only=True` with Ultralytics' known-class allowlist. This protects against local
files shadowing official checkpoint names and rejects unsupported pickle globals. Custom
Ultralytics checkpoints can be supplied directly:

```bash
uv run frame-segment --backend yolo --model path/to/custom.pt \
  --input path/to/frame.jpg --output-dir runs/custom
```

## Try RF-DETR

```bash
uv run frame-segment \
  --backend rfdetr \
  --model small \
  --input path/to/frame.jpg \
  --output-dir runs/rfdetr
```

Supported RF-DETR sizes are `nano`, `small`, `medium`, and `large`. RF-DETR chooses the available PyTorch device automatically.

## Options

```text
--backend {yolo,rfdetr}       Required model family
--model MODEL                 YOLO checkpoint name/path or RF-DETR size
--input PATH                  Required input image
--output-dir PATH             Output directory (default: runs/latest)
--confidence 0..1             Detection threshold (default: 0.25)
--device DEVICE               YOLO device (default: cuda)
```

## Outputs

Each run creates:

- `inference_ms` measures the model adapter call only; output rendering and file I/O are excluded.

```text
runs/<experiment>/
├── overlay.png
├── predictions.json
└── masks/
    ├── 000-person.png
    └── 001-bicycle.png
```

`predictions.json` uses a backend-independent format:

```json
{
  "image": "path/to/frame.jpg",
  "width": 1920,
  "height": 1080,
  "inference_ms": 18.742,
  "detections": [
    {
      "class_id": 0,
      "label": "person",
      "confidence": 0.94,
      "box_xyxy": [120.0, 80.0, 640.0, 1000.0],
      "mask_file": "masks/000-person.png"
    }
  ]
}
```

## Compare models fairly

Use the same input images and confidence threshold. Ignore the first run when measuring speed because it includes checkpoint download, CUDA initialization, and model warm-up. For useful measurements, process a representative image set and report median and 95th-percentile end-to-end latency.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

The unit tests use lightweight fake model results, so they do not download checkpoints or require a GPU.
