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
from core.services import acceptance_matrix
from core.services.acceptance_matrix import (
    AcceptanceMatrixCombinationResult,
    AcceptanceMatrixResult,
    ColorAcceptanceMetrics,
)
from core.services.color_baseline_recalibration import (
    ColorBaselineBuild,
    ColorBaselineCandidateStore,
    ColorBaselineOutlierFilterReport,
)
from core.services.model_acceptance import (
    AcceptanceMetrics,
    AcceptanceRepository,
)
from core.station_data import load_station_data_paths


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


def test_dialog_shows_incompatible_color_baseline_without_letting_it_run(
    tmp_path: Path,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unusable in-scope baseline is visible, explained, and unselectable.

    Hiding it made the tool look broken; showing it must not make it runnable,
    because its crop coordinate space would produce a wrong verdict.
    """
    baselines_root = load_station_data_paths(tmp_path.resolve()).color_baselines
    ColorBaselineCandidateStore(baselines_root).commit(
        product="Cable1",
        area="A",
        model_type="yolo",
        build=ColorBaselineBuild(
            status="INCOMPLETE",
            model_payload={"summary": {"Black": {"count": 30}}},
            report_payload={"status": "INCOMPLETE", "color_reports": []},
            evidence_sha256="e" * 64,
            color_reports=(),
            outlier_filter=ColorBaselineOutlierFilterReport(
                status="NOT_RUN",
                total_sample_count=0,
                z_score_threshold=6.0,
                maximum_auto_exclusion_fraction=0.1,
                candidate_sample_ids=(),
                excluded_sample_ids=(),
                findings=(),
            ),
        ),
    )
    monkeypatch.setattr(acceptance_matrix, "ALGORITHM_VERSION", "stats-robust-v99")

    dialog = _build_dialog(tmp_path, app)

    assert dialog.color_table.rowCount() == 2
    assert "無法使用" in dialog.color_table.item(1, 2).text()
    assert "stats-robust-v99" in dialog.color_table.item(1, 2).text()
    selected = dialog._selected_colors()
    assert [variant.label for variant in selected] == [
        "完整顏色基準（不套用校正修訂）"
    ]
    assert "1 張已確認照片 × 1 組 = 1 次推論" in dialog.workload_label.text()
    assert "尚未建立任何顏色模型" in dialog.color_hint.text()
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


def test_dialog_pairs_every_count_with_its_own_rate(
    tmp_path: Path,
    app: QApplication,
) -> None:
    """Overall and color-only figures must never be read as one metric.

    The counts differ (38 overkills, 36 of them attributable to color) and so
    do the rates, but both share the 173 真實 OK denominator. Rendering the
    overall count beside the color rate let an operator divide one by the
    other, never reproduce the printed percentage, and conclude the tool was
    non-deterministic.
    """
    dialog = _build_dialog(tmp_path, app)
    reference = _combination(
        combination_id="model__color-active",
        color_label="目前正式設定",
        metrics=AcceptanceMetrics(
            confirmed=250,
            pending=0,
            inferred=250,
            tp=77,
            fp=38,
            fn=0,
            tn=135,
            errors=0,
        ),
        color_metrics=ColorAcceptanceMetrics(
            tp=0,
            fp=36,
            fn=0,
            tn=137,
            unknown_truth=77,
            errors=0,
        ),
        changed_from_reference=0,
    )
    candidate = _combination(
        combination_id="model__color-base",
        color_label="完整顏色基準",
        metrics=AcceptanceMetrics(
            confirmed=250,
            pending=0,
            inferred=250,
            tp=77,
            fp=34,
            fn=0,
            tn=139,
            errors=0,
        ),
        color_metrics=ColorAcceptanceMetrics(
            tp=0,
            fp=32,
            fn=0,
            tn=141,
            unknown_truth=77,
            errors=0,
        ),
        changed_from_reference=8,
    )

    dialog._render_results(
        _result(tmp_path, (reference, candidate))
    )

    assert dialog.result_table.item(0, 3).text() == "38（21.97%）"
    assert dialog.result_table.item(0, 5).text() == "36（20.81%）"
    assert dialog.result_table.item(1, 3).text() == "34（19.65%）"
    assert dialog.result_table.item(1, 5).text() == "32（18.50%）"
    dialog.close()


def test_dialog_names_the_reference_row_instead_of_showing_zero_changes(
    tmp_path: Path,
    app: QApplication,
) -> None:
    """The first row is the baseline; a later 0 means it matched the baseline.

    Both used to render as ``0``, which made "is the comparison basis" and
    "produced identical verdicts to the comparison basis" indistinguishable.
    """
    dialog = _build_dialog(tmp_path, app)
    metrics = AcceptanceMetrics(
        confirmed=1,
        pending=0,
        inferred=1,
        tp=0,
        fp=0,
        fn=0,
        tn=1,
        errors=0,
    )
    color_metrics = ColorAcceptanceMetrics(
        tp=0, fp=0, fn=0, tn=1, unknown_truth=0, errors=0
    )

    dialog._render_results(
        _result(
            tmp_path,
            (
                _combination(
                    combination_id="model__first",
                    color_label="內建",
                    metrics=metrics,
                    color_metrics=color_metrics,
                    changed_from_reference=0,
                ),
                _combination(
                    combination_id="model__same",
                    color_label="等效設定",
                    metrics=metrics,
                    color_metrics=color_metrics,
                    changed_from_reference=0,
                ),
            ),
        )
    )

    assert dialog.result_table.item(0, 9).text() == "基準組"
    assert dialog.result_table.item(1, 9).text() == "0"
    dialog.close()


def _combination(
    *,
    combination_id: str,
    color_label: str,
    metrics: AcceptanceMetrics,
    color_metrics: ColorAcceptanceMetrics,
    changed_from_reference: int,
) -> AcceptanceMatrixCombinationResult:
    return AcceptanceMatrixCombinationResult(
        combination_id=combination_id,
        model_variant_id="model",
        model_label="YOLO 1.0.6",
        color_variant_id=combination_id.split("__")[-1],
        color_label=color_label,
        metrics=metrics,
        color_metrics=color_metrics,
        average_latency_ms=100.0,
        p95_latency_ms=110.0,
        changed_from_reference=changed_from_reference,
        changed_sample_ids=(),
    )


def _result(
    tmp_path: Path,
    combinations: tuple[AcceptanceMatrixCombinationResult, ...],
) -> AcceptanceMatrixResult:
    return AcceptanceMatrixResult(
        run_id="run",
        run_root=tmp_path / "run",
        report_path=tmp_path / "run" / "report.json",
        summary_csv_path=tmp_path / "run" / "summary.csv",
        samples_csv_path=tmp_path / "run" / "samples.csv",
        manifest_sha256="abc",
        sample_count=250,
        combinations=combinations,
    )


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
