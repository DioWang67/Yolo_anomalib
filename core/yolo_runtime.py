"""Helpers for selecting a YOLO deployment runtime from model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

YoloRuntime = Literal["pytorch", "openvino", "onnx", "tensorrt", "unknown"]


@dataclass(frozen=True)
class YoloRuntimeInfo:
    """Runtime setup decisions for a YOLO model artifact.

    Args:
        runtime: Detected runtime family.
        setup_device: Device used for PyTorch model placement. Exported
            runtimes are not moved with ``model.to()``.
        predict_device: Device forwarded to Ultralytics prediction calls when
            the runtime needs the inference backend selected per call.
        call_fuse: Whether ``YOLO.fuse()`` should be called during load.
        use_torch_half: Whether to convert the underlying torch module to FP16.
        use_torch_amp: Whether inference should use PyTorch autocast.
        is_exported: True for non-PyTorch deployment artifacts.
    """

    runtime: YoloRuntime
    setup_device: str | None
    predict_device: str | None
    call_fuse: bool
    use_torch_half: bool
    use_torch_amp: bool
    is_exported: bool


def classify_yolo_artifact(weights: str | Path) -> YoloRuntime:
    """Infer the YOLO runtime from a weights path.

    Args:
        weights: File or directory passed to ``ultralytics.YOLO``.

    Returns:
        Runtime label. Unknown files default to ``pytorch`` because existing
        model configs use PyTorch checkpoints.
    """
    path = Path(str(weights))
    suffix = path.suffix.lower()

    if suffix == ".onnx":
        return "onnx"
    if suffix == ".engine":
        return "tensorrt"
    if suffix == ".xml":
        return "openvino"
    if suffix in {".pt", ".pth", ".torchscript"}:
        return "pytorch"

    if path.is_dir():
        if path.name.lower().endswith("_openvino_model"):
            return "openvino"
        try:
            if any(child.suffix.lower() == ".xml" for child in path.iterdir()):
                return "openvino"
        except OSError:
            return "unknown"

    return "pytorch"


def detect_yolo_runtime(
    weights: str | Path,
    device: str | None,
    *,
    cuda_available: bool = False,
) -> YoloRuntimeInfo:
    """Build runtime setup decisions for a YOLO artifact.

    Args:
        weights: YOLO model artifact path.
        device: Configured device string, for example ``cpu``, ``cuda:0`` or
            ``intel:cpu``.
        cuda_available: Whether CUDA is available for PyTorch runtime.

    Returns:
        Runtime information used by ``YOLOInferenceModel``.
    """
    runtime = classify_yolo_artifact(weights)
    normalized_device = _clean_device(device)

    if runtime == "pytorch":
        setup_device = _pytorch_device(normalized_device, cuda_available)
        use_cuda = setup_device != "cpu" and setup_device is not None
        return YoloRuntimeInfo(
            runtime=runtime,
            setup_device=setup_device,
            predict_device=None,
            call_fuse=True,
            use_torch_half=use_cuda,
            use_torch_amp=use_cuda,
            is_exported=False,
        )

    if runtime == "openvino":
        return YoloRuntimeInfo(
            runtime=runtime,
            setup_device=None,
            predict_device=_openvino_predict_device(normalized_device),
            call_fuse=False,
            use_torch_half=False,
            use_torch_amp=False,
            is_exported=True,
        )

    if runtime in {"onnx", "tensorrt"}:
        return YoloRuntimeInfo(
            runtime=runtime,
            setup_device=None,
            predict_device=_generic_export_predict_device(normalized_device),
            call_fuse=False,
            use_torch_half=False,
            use_torch_amp=False,
            is_exported=True,
        )

    return YoloRuntimeInfo(
        runtime=runtime,
        setup_device=None,
        predict_device=_generic_export_predict_device(normalized_device),
        call_fuse=False,
        use_torch_half=False,
        use_torch_amp=False,
        is_exported=True,
    )


def _clean_device(device: str | None) -> str | None:
    if device is None:
        return None
    cleaned = str(device).strip().lower()
    return cleaned or None


def _pytorch_device(device: str | None, cuda_available: bool) -> str:
    if device in {None, "auto"}:
        return "cuda:0" if cuda_available else "cpu"
    return str(device)


def _openvino_predict_device(device: str | None) -> str | None:
    if device in {None, "auto"}:
        return None
    if device.startswith("intel:"):
        return device
    if device in {"cpu", "gpu", "npu"}:
        return f"intel:{device}"
    return device


def _generic_export_predict_device(device: str | None) -> str | None:
    if device in {None, "auto"}:
        return None
    return device
