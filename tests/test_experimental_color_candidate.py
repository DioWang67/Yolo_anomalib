from __future__ import annotations

import csv

import pytest

from tools.color_calibration_service import ColorCalibrationError
from tools.experimental_color_candidate import ExperimentalColorCandidateService


def _feedback(tmp_path, *, ng_count=0):
    path = tmp_path / "feedback.csv"
    fieldnames = (
        "sample_id",
        "item_index",
        "image_sha256",
        "product",
        "area",
        "model_type",
        "checker_type",
        "threshold_key",
        "failure_kind",
        "diff",
        "threshold",
        "actual_is_ok",
    )
    rows = []
    for index in range(30):
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "item_index": "0",
                "image_sha256": f"{index:064x}",
                "product": "Cable1",
                "area": "A",
                "model_type": "yolo",
                "checker_type": "stats",
                "threshold_key": "black",
                "failure_kind": "threshold",
                "diff": f"{0.56 + (index % 7) * 0.01:.2f}",
                "threshold": "0.55",
                "actual_is_ok": "0" if index < ng_count else "1",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _service(tmp_path):
    models = tmp_path / "models"
    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        "color_checker_type: stats\n"
        "color_score_threshold: 0.45\n"
        "color_threshold_overrides:\n"
        "  black: 0.45\n",
        encoding="utf-8",
    )
    return ExperimentalColorCandidateService(
        models_root=models,
        revisions_root=tmp_path / ".color_revisions",
    )


def test_ok_only_candidate_creates_inactive_v102_with_active_v101_baseline(tmp_path):
    feedback = _feedback(tmp_path)
    service = _service(tmp_path)
    eligible = service.eligible_scopes((feedback,))

    assert len(eligible) == 1
    assert eligible[0].scope.threshold_key == "black"
    assert eligible[0].current_public_threshold == pytest.approx(0.55)
    assert eligible[0].proposed_public_threshold == pytest.approx(0.60)
    assert eligible[0].ng_count == 0

    candidate = service.create_candidate(
        (feedback,),
        eligible[0].scope.scope_hash,
        operator="operator-a",
        reason="bounded OK-only experiment",
    )

    assert candidate.baseline_version == "color-v1.0.1"
    assert candidate.candidate_version == "color-v1.0.2"
    assert candidate.active is False
    active = service.revision_store.read_active_pointer(candidate.scope)
    assert active["display_version"] == "color-v1.0.1"
    history = service.revision_store.revision_history(candidate.scope)
    assert [item.display_version for item in history] == [
        "color-v1.0.2",
        "color-v1.0.1",
    ]
    assert history[0].evidence_level == "OK_ONLY"
    assert history[0].active is False


def test_ok_only_candidate_requires_zero_ng_and_supports_activation_and_rollback(
    tmp_path,
):
    ineligible_feedback = _feedback(tmp_path, ng_count=1)
    service = _service(tmp_path)
    assert service.eligible_scopes((ineligible_feedback,)) == ()

    eligible_feedback = _feedback(tmp_path, ng_count=0)
    scope = service.eligible_scopes((eligible_feedback,))[0]
    candidate = service.create_candidate(
        (eligible_feedback,),
        scope.scope.scope_hash,
        operator="operator-a",
        reason="candidate",
    )
    service.activate_candidate(
        candidate,
        operator="reviewer-b",
        reason="limited production trial",
    )
    assert (
        service.revision_store.read_active_pointer(candidate.scope)[
            "display_version"
        ]
        == "color-v1.0.2"
    )

    service.revision_store.rollback(
        candidate.scope,
        "color-v1.0.1",
        operator="reviewer-b",
        reason="trial complete",
    )
    assert (
        service.revision_store.read_active_pointer(candidate.scope)[
            "display_version"
        ]
        == "color-v1.0.1"
    )


def test_ok_only_candidate_rejects_stale_runtime_threshold(tmp_path):
    feedback = _feedback(tmp_path)
    service = _service(tmp_path)
    scope = service.eligible_scopes((feedback,))[0]
    config = (
        tmp_path / "models" / "Cable1" / "A" / "yolo" / "config.yaml"
    )
    config.write_text(
        "color_checker_type: stats\n"
        "color_score_threshold: 0.40\n"
        "color_threshold_overrides:\n"
        "  black: 0.40\n",
        encoding="utf-8",
    )

    with pytest.raises(ColorCalibrationError) as caught:
        service.create_candidate(
            (feedback,),
            scope.scope.scope_hash,
            operator="operator-a",
            reason="stale candidate",
        )

    assert caught.value.code == "CURRENT_CONFIG_STALE"


def test_ok_only_candidate_supports_stats_checker_builtin_threshold(tmp_path):
    feedback = _feedback(tmp_path)
    service = _service(tmp_path)
    config = (
        tmp_path / "models" / "Cable1" / "A" / "yolo" / "config.yaml"
    )
    config.write_text(
        "color_checker_type: stats\n"
        "color_model_path: color_stats.json\n",
        encoding="utf-8",
    )
    scope = service.eligible_scopes((feedback,))[0]

    candidate = service.create_candidate(
        (feedback,),
        scope.scope.scope_hash,
        operator="operator-a",
        reason="default black threshold",
    )

    assert candidate.current_public_threshold == pytest.approx(0.55)
    assert candidate.proposed_public_threshold == pytest.approx(0.60)
