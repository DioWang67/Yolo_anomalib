from __future__ import annotations

import logging

import numpy as np
import pytest

from core.pipeline.context import DetectionContext
from core.pipeline.finalize import finalize_status
from core.pipeline.steps import (
    CountCheckStep,
    CrossClassDuplicateFilterStep,
    SequenceCheckStep,
)
from core.services.cross_class_duplicate_filter import (
    DuplicateFilterConfigurationError,
    DuplicateFilterMode,
    DuplicateFilterPolicy,
    analyze_cross_class_duplicates,
)


def _incident_detections() -> list[dict]:
    """Minimal replay of the Cable1/A 1.0.6 duplicate shown by the operator."""
    return [
        {
            "class": "Orange",
            "verified_class": "Orange",
            "confidence": 0.666983,
            "bbox": [114, 353, 141, 395],
        },
        {
            "class": "Red",
            "verified_class": "Orange",
            "confidence": 0.505362,
            "bbox": [114, 353, 141, 396],
        },
    ]


def _color_items(*, second_measured: bool = True) -> dict[int, dict]:
    """Color items as ``ColorCheckerService`` really emits them for a duplicate.

    Both boxes cover one orange wire, so both measure ``Orange``. The box the
    detector called ``Red`` therefore has ``is_ok=False`` -- a measured color
    can never agree with two different detector classes at once. Asserting
    ``is_ok=True`` on it, as this fixture once did, describes a state the
    service cannot produce and hid that the filter was unreachable.
    """
    return {
        0: {
            "index": 0,
            "best_color": "Orange",
            "is_ok": True,
            "measurement_is_ok": True,
        },
        1: {
            "index": 1,
            "best_color": "Orange",
            "is_ok": False,
            "measurement_is_ok": second_measured,
        },
    }


def _policy(mode: str = "suppress") -> DuplicateFilterPolicy:
    return DuplicateFilterPolicy.from_options(
        {
            "mode": mode,
            "iou_threshold": 0.90,
            "center_distance_ratio_max": 0.10,
            "area_similarity_min": 0.80,
            "require_same_verified_class": True,
            "require_color_check_pass": True,
            "require_different_raw_class": True,
            "require_position_disabled": True,
        }
    )


def test_incident_replay_identifies_lower_confidence_cross_class_box() -> None:
    result = analyze_cross_class_duplicates(
        _incident_detections(),
        _color_items(),
        _policy(),
    )

    assert result["candidate_count"] == 1
    suppression = result["proposed_suppressions"][0]
    assert suppression["kept_index"] == 0
    assert suppression["suppressed_index"] == 1
    assert suppression["verified_class"] == "Orange"
    assert suppression["iou"] == pytest.approx(0.976744, abs=1e-6)


def test_same_verified_color_is_required() -> None:
    detections = _incident_detections()
    detections[1]["verified_class"] = "Red"

    result = analyze_cross_class_duplicates(
        detections,
        _color_items(),
        _policy(),
    )

    assert result["candidate_count"] == 0
    assert result["proposed_suppressions"] == []


def test_unmeasurable_color_blocks_the_pair() -> None:
    """A color that missed its own threshold is no evidence of one object."""
    result = analyze_cross_class_duplicates(
        _incident_detections(),
        _color_items(second_measured=False),
        _policy(),
    )

    assert result["candidate_count"] == 0


def test_detector_disagreement_alone_does_not_block_the_pair() -> None:
    """The defining trait of a cross-class duplicate must not disqualify it.

    Exactly one of two differently-labelled boxes over one wire can agree with
    the measured color, so requiring both to agree made this filter dead code.
    """
    color_items = _color_items()
    assert color_items[1]["is_ok"] is False, "the incident's box really does fail"

    result = analyze_cross_class_duplicates(
        _incident_detections(),
        color_items,
        _policy(),
    )

    assert result["candidate_count"] == 1
    assert result["proposed_suppressions"][0]["suppressed_index"] == 1


def test_legacy_color_payload_without_measurement_field_still_works() -> None:
    """Replays of results persisted before the two verdicts were split."""
    color_items = {
        0: {"index": 0, "best_color": "Orange", "is_ok": True},
        1: {"index": 1, "best_color": "Orange", "is_ok": False},
    }

    result = analyze_cross_class_duplicates(
        _incident_detections(),
        color_items,
        _policy(),
    )

    assert result["candidate_count"] == 0


def test_color_result_must_match_attached_verified_class() -> None:
    color_items = _color_items()
    color_items[1]["best_color"] = "Red"

    result = analyze_cross_class_duplicates(
        _incident_detections(),
        color_items,
        _policy(),
    )

    assert result["candidate_count"] == 0


def _refuted_box_scores_higher() -> tuple[list[dict], dict[int, dict]]:
    """The 2026-08-19 16:03 incident, where confidence pointed the wrong way.

    One orange wire, two boxes: the detector called the higher-scoring one
    ``Red`` (0.808) and the lower-scoring one ``Orange`` (0.664), and the pixels
    measured Orange. Whichever box survives carries its own color verdict, so
    ranking by confidence alone decided the board's fate by coin-flip.
    """
    detections = [
        {"class": "Red", "verified_class": "Orange", "confidence": 0.807896,
         "bbox": [113, 386, 145, 434]},
        {"class": "Orange", "verified_class": "Orange", "confidence": 0.663847,
         "bbox": [113, 385, 145, 433]},
    ]
    color_items = {
        0: {"index": 0, "best_color": "Orange", "is_ok": False,
            "measurement_is_ok": True},
        1: {"index": 1, "best_color": "Orange", "is_ok": True,
            "measurement_is_ok": True},
    }
    return detections, color_items


def test_corroborated_box_outranks_a_higher_confidence_refuted_one() -> None:
    detections, color_items = _refuted_box_scores_higher()

    result = analyze_cross_class_duplicates(detections, color_items, _policy())

    assert result["candidate_count"] == 1
    suppression = result["proposed_suppressions"][0]
    assert suppression["kept_index"] == 1
    assert suppression["kept_raw_class"] == "Orange"
    assert suppression["suppressed_index"] == 0
    assert suppression["suppressed_raw_class"] == "Red"
    # Deliberately keeping the lower-scoring box: the measurement corroborates
    # its class, and evidence outranks a self-reported score.
    assert suppression["kept_confidence"] < suppression["suppressed_confidence"]


def test_keeping_the_corroborated_box_clears_the_color_verdict() -> None:
    """The payoff: no spurious color FAIL survives on the winner."""
    detections, color_items = _refuted_box_scores_higher()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    context = DetectionContext(
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=frame,
        processed_image=frame,
        result={"detections": detections},
        status="DETECTION_FAIL",
        color_result={
            "is_ok": False,
            "status": "evaluated",
            "items": list(color_items.values()),
        },
        config=_PositionConfig(False),
    )

    CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options={**_policy().to_dict(), "enabled": True},
    ).run(context)

    assert context.result["detections"][0]["class"] == "Orange"
    assert context.color_result["is_ok"] is True
    finalize_status(context)
    assert context.status == "PASS"


def test_geometry_guard_rejects_boxes_with_insufficient_overlap() -> None:
    detections = _incident_detections()
    detections[1]["bbox"] = [120, 353, 147, 396]

    result = analyze_cross_class_duplicates(
        detections,
        _color_items(),
        _policy(),
    )

    assert result["candidate_count"] == 0


def test_policy_rejects_unsafe_values() -> None:
    with pytest.raises(DuplicateFilterConfigurationError, match="iou_threshold"):
        DuplicateFilterPolicy.from_options({"iou_threshold": 0.0})
    with pytest.raises(DuplicateFilterConfigurationError, match="mode"):
        DuplicateFilterPolicy.from_options({"mode": "automatic"})
    with pytest.raises(
        DuplicateFilterConfigurationError,
        match="require_color_check_pass",
    ):
        DuplicateFilterPolicy.from_options({"require_color_check_pass": "yes"})


class _PositionConfig:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def is_position_check_enabled(self, product: str, area: str) -> bool:
        return self.enabled


def _context(*, position_enabled: bool = False) -> DetectionContext:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return DetectionContext(
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=frame,
        processed_image=frame,
        result={"detections": _incident_detections()},
        status="DETECTION_FAIL",
        color_result={"is_ok": True, "items": list(_color_items().values())},
        config=_PositionConfig(position_enabled),
    )


def test_suppression_retracts_the_color_failure_it_removed() -> None:
    """The removed box must not keep failing the board it is no longer part of."""
    context = _context()
    context.color_result["is_ok"] = False
    context.color_result["status"] = "evaluated"
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options={**_policy().to_dict(), "enabled": True},
    )

    step.run(context)

    assert context.color_result["is_ok"] is True
    # The evidence is kept for review even though it no longer votes.
    assert len(context.color_result["items"]) == 2
    finalize_status(context)
    assert context.status == "PASS"


def test_suppression_keeps_a_failure_that_belongs_to_a_surviving_box() -> None:
    context = _context()
    context.color_result["is_ok"] = False
    context.color_result["status"] = "evaluated"
    context.color_result["items"][0]["is_ok"] = False
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options={**_policy().to_dict(), "enabled": True},
    )

    step.run(context)

    assert context.color_result["is_ok"] is False
    finalize_status(context)
    assert context.status == "DETECTION_FAIL"


def test_suppression_does_not_reinterpret_an_unevaluated_color_check() -> None:
    """A check that measured nothing has no verdict to recompute."""
    context = _context()
    context.color_result["is_ok"] = False
    context.color_result["status"] = "no_detections"
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options={**_policy().to_dict(), "enabled": True},
    )

    step.run(context)

    assert context.color_result["is_ok"] is False


def test_suppress_mode_preserves_raw_and_updates_effective_detections() -> None:
    context = _context()
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options={**_policy().to_dict(), "enabled": True},
    )

    step.run(context)

    assert len(context.result["raw_detections"]) == 2
    assert len(context.result["detections"]) == 1
    assert context.result["detections"][0]["source_index"] == 0
    metadata = context.result["duplicate_filter"]
    assert metadata["status"] == "suppressed"
    assert metadata["suppressed_count"] == 1
    assert metadata["effective_count"] == 1


def test_report_only_mode_does_not_change_detections_or_verdict() -> None:
    context = _context()
    original_status = context.status
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options={**_policy(DuplicateFilterMode.REPORT_ONLY.value).to_dict()},
    )

    step.run(context)

    assert len(context.result["detections"]) == 2
    assert "raw_detections" not in context.result
    assert context.status == original_status
    assert context.result["duplicate_filter"]["status"] == "reported"
    assert context.result["duplicate_filter"]["suppressed_count"] == 0
    assert context.result["duplicate_filter"]["would_suppress_count"] == 1


def test_position_enabled_blocks_suppression_fail_closed() -> None:
    context = _context(position_enabled=True)
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options=_policy().to_dict(),
    )

    step.run(context)

    assert len(context.result["detections"]) == 2
    assert "raw_detections" not in context.result
    assert (
        context.result["duplicate_filter"]["status"]
        == "blocked_position_enabled"
    )


@pytest.mark.parametrize("color_result", [{}, {"items": []}])
def test_missing_or_empty_color_items_block_before_duplicate_analysis(
    color_result,
) -> None:
    context = _context()
    context.color_result = color_result
    step = CrossClassDuplicateFilterStep(
        logging.getLogger(__name__),
        options=_policy().to_dict(),
    )

    step.run(context)

    metadata = context.result["duplicate_filter"]
    assert metadata["status"] == "blocked_color_result_unavailable"
    assert metadata["candidate_count"] == 0
    assert metadata["would_suppress_count"] == 0
    assert "raw_detections" not in context.result


class _PipelineConfig(_PositionConfig):
    def get_items_by_area(self, product: str, area: str) -> list[str]:
        return ["Red", "Green", "Orange", "Yellow", "Black", "Black"]


def test_pipeline_recomputes_count_and_sequence_from_effective_detections() -> None:
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    detections = [
        {"class": "Red", "verified_class": "Red", "confidence": 0.95, "bbox": [10, 50, 20, 80]},
        {"class": "Green", "verified_class": "Green", "confidence": 0.94, "bbox": [30, 50, 40, 80]},
        {"class": "Orange", "verified_class": "Orange", "confidence": 0.90, "bbox": [50, 50, 60, 80]},
        {"class": "Red", "verified_class": "Orange", "confidence": 0.50, "bbox": [50, 50, 60, 80]},
        {"class": "Yellow", "verified_class": "Yellow", "confidence": 0.93, "bbox": [70, 50, 80, 80]},
        {"class": "Black", "verified_class": "Black", "confidence": 0.92, "bbox": [90, 50, 100, 80]},
        {"class": "Black", "verified_class": "Black", "confidence": 0.91, "bbox": [110, 50, 120, 80]},
    ]
    color_items = [
        {
            "index": index,
            "best_color": detection["verified_class"],
            "is_ok": True,
        }
        for index, detection in enumerate(detections)
    ]
    context = DetectionContext(
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=frame,
        processed_image=frame,
        result={
            "detections": detections,
            "missing_items": [],
            "unexpected_items": ["Orange"],
        },
        status="DETECTION_FAIL",
        color_result={"is_ok": True, "items": color_items},
        config=_PipelineConfig(False),
    )
    logger = logging.getLogger(__name__)

    CrossClassDuplicateFilterStep(logger, options=_policy().to_dict()).run(context)
    CountCheckStep(
        logger,
        product="Cable1",
        area="A",
        options={"enabled": True, "strict": True},
    ).run(context)
    SequenceCheckStep(
        logger,
        product="Cable1",
        area="A",
        options={
            "enabled": True,
            "expected": ["Red", "Green", "Orange", "Yellow", "Black", "Black"],
        },
    ).run(context)
    finalize_status(context)

    assert len(context.result["raw_detections"]) == 7
    assert len(context.result["detections"]) == 6
    assert context.result["count_check"]["is_ok"] is True
    assert context.result["unexpected_items"] == []
    assert context.result["sequence_check"]["is_ok"] is True
    assert context.status == "PASS"
