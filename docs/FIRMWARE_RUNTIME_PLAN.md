# Firmware Runtime Optimization Plan

## Scope

This project should keep training and validation in Python, but firmware or
edge runtime should consume exported artifacts whenever possible.

## Task Classification

Class C: CV inference/deployment tooling. The goal is fast iteration with
repeatable measurements, not a full production platform rewrite.

## Recommended Runtime Split

| Model | Development format | Deployment candidate | First target |
| --- | --- | --- | --- |
| YOLO | `.pt` | OpenVINO directory or ONNX | OpenVINO on Intel CPU/iGPU/NPU |
| Anomalib | `.ckpt` Lightning checkpoint | OpenVINO/ONNX only after export validation | Keep Python runtime for now |

YOLO is the safer first optimization target because Ultralytics can load
exported artifacts through the same high-level API. Anomalib should not be
moved into firmware as a Python/Lightning dependency; firmware should receive
only an exported model and fixed post-processing logic.

## Repository Boundary

`yolo11_inference` is the GUI/runtime consumer. It should load and benchmark
artifacts that were produced elsewhere, but it should not own the training-time
export pipeline.

YOLO export and model bundle creation belong in `Yolo11_auto_train`, where the
training run, class names, detection config and position config are available
in one place. This keeps the GUI project focused on inference behavior and
deployment validation.

## Runtime Tooling

Export YOLO from the training project:

```powershell
cd D:\Git\robotlearning\Yolo11_auto_train
picture-tool-pipeline --config configs\<product>.yaml --tasks yolo_train,deploy
```

Benchmark PyTorch YOLO:

```powershell
python tools\runtime_benchmark.py `
  --backend yolo `
  --model models\Cable1\A\yolo\weights\best.pt `
  --images path\to\test_images `
  --device cpu `
  --imgsz 640 `
  --runs 50 `
  --output-json reports\yolo_pt_benchmark.json
```

Benchmark OpenVINO YOLO:

```powershell
python tools\runtime_benchmark.py `
  --backend yolo `
  --model models\Cable1\A\yolo\weights\best_openvino_model `
  --images path\to\test_images `
  --device intel:cpu `
  --imgsz 640 `
  --runs 50 `
  --output-json reports\yolo_openvino_benchmark.json
```

Benchmark current Anomalib Lightning runtime:

```powershell
python tools\runtime_benchmark.py `
  --backend anomalib-lightning `
  --config models\PCBA1\A\anomalib\config.yaml `
  --product PCBA1 `
  --area A `
  --images path\to\test_images `
  --runs 20 `
  --output-json reports\anomalib_lightning_benchmark.json
```

## Acceptance Criteria Before Firmware Use

1. YOLO OpenVINO and PyTorch results agree on PASS/FAIL for the validation set.
2. Bounding boxes remain within the accepted tolerance for production checks.
3. Anomalib exported runtime reproduces anomaly score ordering and PASS/FAIL
   decisions against the current Lightning baseline.
4. Runtime package size, cold-start time and peak memory are measured.
5. Firmware-side preprocessing and post-processing are frozen and documented.

## Trade-off

The current branch does not force OpenVINO everywhere. It adds runtime
detection and benchmark tooling first, because a firmware target can vary
between Intel, NVIDIA Jetson, ARM SoC, or vendor NPU. Hard-coding OpenVINO
before measuring would make non-Intel deployments worse.
