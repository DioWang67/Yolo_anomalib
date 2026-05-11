from __future__ import annotations

from core.yolo_runtime import classify_yolo_artifact, detect_yolo_runtime


def test_classify_yolo_artifact_detects_common_export_formats(tmp_path):
    openvino_dir = tmp_path / "best_openvino_model"
    openvino_dir.mkdir()
    (openvino_dir / "best.xml").write_text("<xml/>", encoding="utf-8")

    assert classify_yolo_artifact("best.pt") == "pytorch"
    assert classify_yolo_artifact("best.onnx") == "onnx"
    assert classify_yolo_artifact("best.engine") == "tensorrt"
    assert classify_yolo_artifact(openvino_dir) == "openvino"
    assert classify_yolo_artifact(openvino_dir / "best.xml") == "openvino"


def test_detect_yolo_runtime_uses_cuda_for_pytorch_auto_device():
    info = detect_yolo_runtime("best.pt", "auto", cuda_available=True)

    assert info.runtime == "pytorch"
    assert info.setup_device == "cuda:0"
    assert info.predict_device is None
    assert info.call_fuse is True
    assert info.use_torch_half is True
    assert info.use_torch_amp is True


def test_detect_yolo_runtime_maps_openvino_cpu_to_intel_device(tmp_path):
    openvino_dir = tmp_path / "best_openvino_model"
    openvino_dir.mkdir()

    info = detect_yolo_runtime(openvino_dir, "cpu", cuda_available=True)

    assert info.runtime == "openvino"
    assert info.setup_device is None
    assert info.predict_device == "intel:cpu"
    assert info.call_fuse is False
    assert info.use_torch_half is False
    assert info.use_torch_amp is False


def test_detect_yolo_runtime_keeps_onnx_as_exported_runtime():
    info = detect_yolo_runtime("best.onnx", "auto")

    assert info.runtime == "onnx"
    assert info.setup_device is None
    assert info.predict_device is None
    assert info.is_exported is True
