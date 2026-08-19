"""Export human-reviewed color failures into an isolated calibration dataset."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.review_routing import action_route, color_failure_items

COLOR_FEEDBACK_FIELDS = (
    "sample_id",
    "image_sha256",
    "product",
    "area",
    "model_type",
    "timestamp",
    "source_manifest",
    "config_snapshot_path",
    "source_image",
    "output_image",
    "review_label",
    "product_verdict",
    "detection_verdict",
    "color_verdict",
    "action_route",
    "checker_type",
    "item_index",
    "expected_color",
    "predicted_color",
    "threshold_key",
    "failure_kind",
    "diff",
    "threshold",
    "score",
    "runtime_is_ok",
    "actual_is_ok",
    "model_version",
    "review_note",
)


@dataclass(frozen=True)
class ColorFeedbackExportReport:
    """Summary of a color-feedback export operation."""

    item_count: int
    case_count: int
    targets: tuple[tuple[str, str], ...]
    manifest_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ColorFeedbackScopeProgress:
    """Persisted calibration evidence counts for one threshold scope."""

    product: str
    area: str
    model_type: str
    checker_type: str
    threshold_key: str
    total_count: int
    ok_count: int
    ng_count: int


@dataclass(frozen=True)
class _PlannedFeedback:
    target_root: Path
    source_image: Path
    output_image: Path
    values: dict[str, str]


def export_color_feedback(
    review_rows: Iterable[dict[str, str]],
    *,
    source_manifest: str | Path,
    output_root: str | Path,
) -> ColorFeedbackExportReport:
    """Copy selected color cases and atomically merge item-level feedback.

    Color-only decisions are intentionally stored under ``color_review`` and
    never materialized under YOLO ``raw/images`` or ``raw/labels``.
    """
    manifest_path = Path(source_manifest).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    planned: list[_PlannedFeedback] = []
    case_ids: set[str] = set()

    for row in review_rows:
        route = action_route(row)
        if route not in {"color", "both"}:
            continue
        if str(row.get("training_selected") or "1") == "0":
            continue
        color_verdict = _normalized_color_verdict(row)
        failed_items = color_failure_items(row)
        if not failed_items:
            raise ValueError(
                "Color review cannot be exported because the saved snapshot "
                "contains no failed color item. Re-run the inspection or exclude it."
            )
        source_image = _source_image(row)
        if source_image is None:
            raise ValueError(
                "Color review source image is missing: "
                f"{row.get('product', '')}/{row.get('area', '')} "
                f"{row.get('timestamp', '')}"
            )
        image_sha256 = _sha256_file(source_image)
        product = str(row.get("product") or "unknown")
        area = str(row.get("area") or "unknown")
        sample_id = _sample_id(product, area, image_sha256)
        target_root = destination / _safe_name(product) / _safe_name(area)
        output_image = (
            target_root
            / "color_review"
            / "images"
            / f"{sample_id}{source_image.suffix.lower() or '.png'}"
        )
        case_ids.add(sample_id)
        for fallback_index, item in enumerate(failed_items):
            item_index = _item_index(item, fallback_index)
            diff = _finite_float(item.get("diff"), field="diff")
            threshold = _finite_float(item.get("threshold"), field="threshold")
            expected_color = str(
                item.get("class_name") or item.get("class") or ""
            ).strip()
            predicted_color = str(item.get("best_color") or "").strip()
            threshold_key = predicted_color or expected_color or "global"
            failure_kind = "threshold" if diff > threshold else "rule"
            planned.append(
                _PlannedFeedback(
                    target_root=target_root,
                    source_image=source_image,
                    output_image=output_image,
                    values={
                        "sample_id": sample_id,
                        "image_sha256": image_sha256,
                        "product": product,
                        "area": area,
                        "model_type": str(row.get("detector") or "yolo").lower(),
                        "timestamp": str(row.get("timestamp") or ""),
                        "source_manifest": str(manifest_path),
                        "config_snapshot_path": str(
                            row.get("config_snapshot_path") or ""
                        ),
                        "source_image": str(source_image.resolve()),
                        "output_image": str(output_image.resolve()),
                        "review_label": str(row.get("review_label") or ""),
                        "product_verdict": str(row.get("product_verdict") or ""),
                        "detection_verdict": str(
                            row.get("detection_verdict") or "correct"
                        ),
                        "color_verdict": color_verdict,
                        "action_route": route,
                        "checker_type": str(
                            row.get("color_checker_type") or "enhanced"
                        ).lower(),
                        "item_index": str(item_index),
                        "expected_color": expected_color,
                        "predicted_color": predicted_color,
                        "threshold_key": threshold_key.lower(),
                        "failure_kind": failure_kind,
                        "diff": _format_float(diff),
                        "threshold": _format_float(threshold),
                        "score": _format_float(max(0.0, 1.0 - diff)),
                        "runtime_is_ok": "0",
                        "actual_is_ok": (
                            "1" if color_verdict == "actually_ok" else "0"
                        ),
                        "model_version": str(row.get("model_version") or ""),
                        "review_note": str(row.get("review_note") or ""),
                    },
                )
            )

    if not planned:
        return ColorFeedbackExportReport(0, 0, (), ())

    created_images: list[Path] = []
    manifests: list[Path] = []
    manifest_originals: dict[Path, bytes | None] = {}
    try:
        for image_plan in _unique_image_plans(planned):
            if image_plan.output_image.is_file():
                continue
            image_plan.output_image.parent.mkdir(parents=True, exist_ok=True)
            temporary = image_plan.output_image.with_name(
                f".{image_plan.output_image.name}.{os.getpid()}.tmp"
            )
            try:
                shutil.copy2(image_plan.source_image, temporary)
                temporary.replace(image_plan.output_image)
                created_images.append(image_plan.output_image)
            finally:
                temporary.unlink(missing_ok=True)

        by_target: dict[Path, list[dict[str, str]]] = {}
        for item in planned:
            by_target.setdefault(item.target_root, []).append(item.values)
        for target_root, new_rows in sorted(
            by_target.items(), key=lambda value: str(value[0])
        ):
            feedback_path = target_root / "color_review" / "feedback.csv"
            manifest_originals[feedback_path] = (
                feedback_path.read_bytes() if feedback_path.is_file() else None
            )
            existing = _read_feedback(feedback_path)
            merged = _merge_feedback(existing, new_rows)
            _write_feedback_atomic(feedback_path, merged)
            manifests.append(feedback_path)
            _verify_feedback_rows(feedback_path, new_rows)
        for image_plan in _unique_image_plans(planned):
            if not image_plan.output_image.is_file():
                raise OSError(
                    f"Color feedback image was not persisted: {image_plan.output_image}"
                )
            if _sha256_file(image_plan.output_image) != image_plan.values["image_sha256"]:
                raise OSError(
                    f"Color feedback image verification failed: {image_plan.output_image}"
                )
    except (OSError, shutil.Error, csv.Error):
        for path in reversed(manifests):
            original = manifest_originals[path]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                _write_bytes_atomic(path, original)
        for path in reversed(created_images):
            path.unlink(missing_ok=True)
        raise

    targets = tuple(
        sorted({(item.values["product"], item.values["area"]) for item in planned})
    )
    return ColorFeedbackExportReport(
        item_count=len(planned),
        case_count=len(case_ids),
        targets=targets,
        manifest_paths=tuple(manifests),
    )


def read_color_feedback_progress(
    manifest_paths: Sequence[str | Path],
) -> tuple[ColorFeedbackScopeProgress, ...]:
    """Read back and summarize durable feedback per calibration scope."""
    rows_by_identity: dict[tuple[str, str], dict[str, str]] = {}
    for raw_path in manifest_paths:
        path = Path(raw_path)
        rows = _read_feedback(path)
        if not path.is_file():
            raise OSError(f"Color feedback manifest is missing: {path}")
        for row in rows:
            identity = (
                str(row.get("sample_id") or ""),
                str(row.get("item_index") or ""),
            )
            if not all(identity):
                raise ValueError(f"Color feedback identity is incomplete: {path}")
            if str(row.get("actual_is_ok") or "") not in {"0", "1"}:
                raise ValueError(f"Color feedback truth value is invalid: {path}")
            rows_by_identity[identity] = row

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows_by_identity.values():
        scope = tuple(
            str(row.get(field) or "").strip().lower()
            for field in (
                "product",
                "area",
                "model_type",
                "checker_type",
                "threshold_key",
            )
        )
        if not all(scope):
            raise ValueError("Color feedback calibration scope is incomplete")
        grouped.setdefault(scope, []).append(row)

    return tuple(
        ColorFeedbackScopeProgress(
            product=scope[0],
            area=scope[1],
            model_type=scope[2],
            checker_type=scope[3],
            threshold_key=scope[4],
            total_count=len(rows),
            ok_count=sum(row["actual_is_ok"] == "1" for row in rows),
            ng_count=sum(row["actual_is_ok"] == "0" for row in rows),
        )
        for scope, rows in sorted(grouped.items())
    )


def _normalized_color_verdict(row: dict[str, str]) -> str:
    verdict = str(row.get("color_verdict") or "").strip().lower()
    if verdict in {"confirmed_ng", "actually_ok"}:
        return verdict
    label = str(row.get("review_label") or "").strip().lower()
    if label == "color_confirmed_ng":
        return "confirmed_ng"
    if label == "color_false_reject":
        return "actually_ok"
    raise ValueError(f"Unsupported color verdict: {verdict or label}")


def _source_image(row: dict[str, str]) -> Path | None:
    for field in ("preprocessed_path", "original_path"):
        path = Path(str(row.get(field) or ""))
        if path.is_file():
            return path
    return None


def _item_index(item: dict[str, Any], fallback: int) -> int:
    try:
        value = int(item.get("index", fallback))
    except (TypeError, ValueError):
        return fallback
    return value if value >= -1 else fallback


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Color feedback {field} must be numeric") from exc
    if not (float("-inf") < result < float("inf")) or result < 0:
        raise ValueError(f"Color feedback {field} must be finite and non-negative")
    return result


def _format_float(value: float) -> str:
    return f"{value:.9g}"


def _safe_name(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value.strip()
    )
    return safe.strip("._") or "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_id(product: str, area: str, image_sha256: str) -> str:
    raw = f"{product}\0{area}\0{image_sha256}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _unique_image_plans(
    planned: list[_PlannedFeedback],
) -> list[_PlannedFeedback]:
    unique: dict[Path, _PlannedFeedback] = {}
    for item in planned:
        unique[item.output_image] = item
    return list(unique.values())


def _read_feedback(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _merge_feedback(
    existing: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for row in [*existing, *new_rows]:
        key = (str(row.get("sample_id") or ""), str(row.get("item_index") or ""))
        if all(key):
            merged[key] = {
                field: str(row.get(field) or "") for field in COLOR_FEEDBACK_FIELDS
            }
    return [merged[key] for key in sorted(merged)]


def _verify_feedback_rows(path: Path, expected_rows: list[dict[str, str]]) -> None:
    """Fail closed unless every newly exported row survives a disk read-back."""
    persisted = {
        (str(row.get("sample_id") or ""), str(row.get("item_index") or "")): row
        for row in _read_feedback(path)
    }
    expected = {
        (str(row["sample_id"]), str(row["item_index"])): row
        for row in _merge_feedback([], expected_rows)
    }
    for identity, expected_row in expected.items():
        actual_row = persisted.get(identity)
        if actual_row is None or any(
            str(actual_row.get(field) or "") != str(expected_row.get(field) or "")
            for field in COLOR_FEEDBACK_FIELDS
        ):
            raise OSError(
                "Color feedback manifest verification failed for "
                f"{identity[0]}/{identity[1]} in {path}"
            )


def _write_feedback_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLOR_FEEDBACK_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.rollback.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
