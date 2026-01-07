from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.exceptions import BackendInitializationError, BackendNotAvailableError, ModelInitializationError
from core.inference_engine import InferenceEngine


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.enable_yolo = True
    config.enable_anomalib = True
    config.backends = {}
    return config

@patch("core.inference_engine.YOLOInferenceModel")
def test_lazy_initialization_yolo(MockYOLO, mock_config):
    """測試 YOLO 後端的延遲加載"""
    engine = InferenceEngine(mock_config)
    assert "yolo" not in engine.models

    mock_model_inst = MockYOLO.return_value

    # First call triggers init
    engine.infer(np.zeros((10, 10, 3), dtype=np.uint8), "P", "A", "yolo")

    assert "yolo" in engine.models
    MockYOLO.assert_called_once_with(mock_config)
    mock_model_inst.initialize.assert_called_once_with(product="P", area="A")

@patch("core.inference_engine.AnomalibInferenceModel")
def test_lazy_initialization_anomalib(MockAnomalib, mock_config):
    """測試 Anomalib 後端的延遲加載"""
    engine = InferenceEngine(mock_config)
    assert "anomalib" not in engine.models

    mock_model_inst = MockAnomalib.return_value

    # First call triggers init
    engine.infer(np.zeros((10, 10, 3), dtype=np.uint8), "P", "A", "anomalib")

    assert "anomalib" in engine.models
    MockAnomalib.assert_called_once_with(mock_config)
    mock_model_inst.initialize.assert_called_once_with(product="P", area="A")

def test_infer_backend_not_enabled(mock_config):
    """測試請求未啟用的後端"""
    mock_config.enable_yolo = False
    engine = InferenceEngine(mock_config)

    with pytest.raises(BackendNotAvailableError):
        engine.infer(np.zeros((10, 10, 3)), "P", "A", "yolo")

@patch("core.inference_engine.YOLOInferenceModel")
def test_backend_init_failure_propagation(MockYOLO, mock_config):
    """測試後端初始化失敗時的異常傳遞"""
    engine = InferenceEngine(mock_config)

    mock_model_inst = MockYOLO.return_value
    mock_model_inst.initialize.side_effect = ModelInitializationError("Init failed")

    with pytest.raises(BackendInitializationError):
        engine.infer(np.zeros((10, 10, 3)), "P", "A", "yolo")

def test_dynamic_backend_loading(mock_config):
    """測試通過 class_path 動態加載自定義後端"""
    mock_config.backends = {
        "custom": {
            "enabled": True,
            "class_path": "unittest.mock.MagicMock"
        }
    }
    engine = InferenceEngine(mock_config)

    # We need to mock the import or use a real path that exists
    with patch("core.inference_engine.import_module") as mock_import:
        mock_cls = MagicMock()
        mock_import.return_value = MagicMock(MyCustomModel=mock_cls)
        mock_config.backends["custom"]["class_path"] = "fake_module.MyCustomModel"

        engine.infer(np.zeros((10, 10, 3)), "P", "A", "custom")

        assert "custom" in engine.models
        mock_import.assert_called_with("fake_module")

def test_shutdown_cleans_up_models(mock_config):
    """測試 shutdown 是否釋放所有模型的資源"""
    engine = InferenceEngine(mock_config)
    m1 = MagicMock()
    m2 = MagicMock()
    engine.models = {"m1": m1, "m2": m2}

    engine.shutdown()

    m1.shutdown.assert_called_once()
    m2.shutdown.assert_called_once()
    assert len(engine.models) == 0
