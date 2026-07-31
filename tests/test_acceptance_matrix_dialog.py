from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from app.acceptance.matrix_dialog import AcceptanceMatrixDialog
from core.services.acceptance_matrix import (
    AcceptanceMatrixCombinationResult,
    AcceptanceMatrixResult,
    ColorAcceptanceMetrics,
)
from core.services.model_acceptance import (
    AcceptanceMetrics,
    AcceptanceRepository,
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _build_dialog(
    tmp_path: Path, _app: QApplication
) -> AcceptanceMatrixDialog:
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "placeholder.onnx").write_bytes(b"model")
    (model_dir / "config.yaml").write_text(
        "enable_yolo: true\nweights: placeholder.onnx\n",
        encoding="utf-8",
    )
    repository = AcceptanceRepository(tmp_path / "acceptance" / "Cable1" / "A")
    image_path = tmp_path / "sample.png"
    assert cv2.imwrite(
        str(image_path),
        np.full((8, 8, 3), 100, dtype=np.uint8),
    )
    record = repository.import_images(
        (image_path,),
        product="Cable1",
        area="A",
    )[0]
    repository.confirm(
        record.sample_id,
        verdict="OK",
        reviewed_by="reviewer",
    )
    dialog = AcceptanceMatrixDialog(
        project_root=tmp_path,
        repository=repository,
        product="Cable1",
        area="A",
        inference_type="yolo",
    )
    return dialog


def test_dialog_loads_default_cartesian_selection(
    tmp_path: Path, app: QApplication
) -> None:
    dialog = _build_dialog(tmp_path, app)

    assert dialog.model_table.rowCount() == 1
    assert dialog.color_table.rowCount() == 1
    assert dialog._selected_models()[0].label == "目前推論專案模型"
    assert (
        dialog._selected_colors()[0].label
        == "完整顏色基準（不套用校正修訂）"
    )
    assert "1 張已確認照片 × 1 組 = 1 次推論" in dialog.workload_label.text()
    dialog.close()


def test_dialog_marks_unverifiable_color_escape_as_unknown(
    tmp_path: Path,
    app: QApplication,
) -> None:
    dialog = _build_dialog(tmp_path, app)
    combination = AcceptanceMatrixCombinationResult(
        combination_id="model__color",
        model_variant_id="model",
        model_label="YOLO v1",
        color_variant_id="color",
        color_label="Black / color-v1.0.2",
        metrics=AcceptanceMetrics(
            confirmed=1,
            pending=0,
            inferred=1,
            tp=0,
            fp=0,
            fn=0,
            tn=1,
            errors=0,
        ),
        color_metrics=ColorAcceptanceMetrics(
            tp=0,
            fp=0,
            fn=0,
            tn=1,
            unknown_truth=0,
            errors=0,
        ),
        average_latency_ms=12.0,
        p95_latency_ms=12.0,
        changed_from_reference=0,
        changed_sample_ids=(),
    )
    result = AcceptanceMatrixResult(
        run_id="run",
        run_root=tmp_path / "run",
        report_path=tmp_path / "run" / "report.json",
        summary_csv_path=tmp_path / "run" / "summary.csv",
        samples_csv_path=tmp_path / "run" / "samples.csv",
        manifest_sha256="abc",
        sample_count=1,
        combinations=(combination,),
    )

    dialog._render_results(result)

    assert (
        dialog.result_table.item(0, 6).text()
        == "UNKNOWN（無真顏色 NG）"
    )
    dialog.close()


def test_dialog_runs_matrix_in_worker_and_renders_result(
    tmp_path: Path,
    app: QApplication,
) -> None:
    dialog = _build_dialog(tmp_path, app)
    run_root = tmp_path / "report"
    combination = AcceptanceMatrixCombinationResult(
        combination_id="model__color",
        model_variant_id="model",
        model_label="YOLO test",
        color_variant_id="color",
        color_label="內建",
        metrics=AcceptanceMetrics(
            confirmed=1,
            pending=0,
            inferred=1,
            tp=0,
            fp=0,
            fn=0,
            tn=1,
            errors=0,
        ),
        color_metrics=ColorAcceptanceMetrics(
            tp=0,
            fp=0,
            fn=0,
            tn=1,
            unknown_truth=0,
            errors=0,
        ),
        average_latency_ms=8.0,
        p95_latency_ms=8.0,
        changed_from_reference=0,
        changed_sample_ids=(),
    )
    expected = AcceptanceMatrixResult(
        run_id="run",
        run_root=run_root,
        report_path=run_root / "report.json",
        summary_csv_path=run_root / "summary.csv",
        samples_csv_path=run_root / "samples.csv",
        manifest_sha256="abc",
        sample_count=1,
        combinations=(combination,),
    )

    def fake_runner(
        _request,
        *,
        progress_callback,
        cancel_callback,
    ):
        assert not cancel_callback()
        progress_callback(1, 1, "YOLO test × 內建", "sample-1")
        return expected

    dialog._runner = fake_runner
    dialog._start()
    deadline = time.monotonic() + 3.0
    while dialog._worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()

    assert dialog._worker is None
    assert dialog.progress.value() == 1
    assert dialog.result_table.rowCount() == 1
    assert dialog.open_report_button.isEnabled()
    assert str(run_root) in dialog.progress_detail.text()
    dialog.close()
