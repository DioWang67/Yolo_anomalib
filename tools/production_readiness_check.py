"""Validate whether an inference config is ready for controlled production use."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReadinessCheck:
    """One production readiness check result."""

    name: str
    status: str
    message: str


def run_readiness_checks(
    config_path: str | Path,
    *,
    product: str | None = None,
    area: str | None = None,
    source_paths: set[Path] | None = None,
) -> list[ReadinessCheck]:
    """Run production readiness checks against a YAML config.

    Args:
        config_path: Global or model-specific inference config.
        product: Product override. Falls back to ``current_product``.
        area: Area override. Falls back to ``current_area``.
        source_paths: Optional collector populated with every resolved file path
            used by the checks. Callers writing a report must protect these paths
            from destination collisions.

    Returns:
        Ordered readiness check results.
    """
    path = Path(config_path)
    config, effective_path = _load_effective_config(path, product, area)
    if source_paths is not None:
        source_paths.update(
            {
                path.expanduser().resolve(strict=False),
                effective_path.expanduser().resolve(strict=False),
            }
        )
    product_name = product or str(config.get("current_product") or "")
    area_name = area or str(config.get("current_area") or "")
    checks: list[ReadinessCheck] = []

    _add(checks, "config_exists", path.exists(), f"config={path}")
    _add(
        checks,
        "model_config_loaded",
        effective_path.exists(),
        f"effective_config={effective_path}",
    )
    _add(
        checks,
        "product_area",
        bool(product_name and area_name),
        f"product={product_name or '-'}, area={area_name or '-'}",
    )

    weights = str(config.get("weights") or "")
    weights_path = _resolve_existing_path(effective_path.parent, weights) if weights else None
    if source_paths is not None and weights_path is not None:
        source_paths.add(weights_path.expanduser().resolve(strict=False))
    _add(checks, "weights_configured", bool(weights), "weights path is configured")
    _add(
        checks,
        "weights_exists",
        bool(weights_path and weights_path.exists()),
        f"weights={weights_path or '-'}",
    )

    expected_items = _expected_items(config, product_name, area_name)
    _add(checks, "expected_items", bool(expected_items), f"expected item count={len(expected_items)}")

    position_cfg = _position_config(config, product_name, area_name)
    position_required = _position_required(config, position_cfg)
    position_enabled = position_cfg.get("enabled", False)
    _add(
        checks,
        "position_check_enabled",
        not position_required or position_enabled is True,
        "position validation enabled"
        if position_required and isinstance(position_enabled, bool)
        else "position_config.enabled must be a YAML boolean"
        if position_required
        else "position validation is not declared in defect coverage",
    )
    expected_boxes = position_cfg.get("expected_boxes") if isinstance(position_cfg, dict) else {}
    expected_boxes = expected_boxes if isinstance(expected_boxes, dict) else {}
    _add(
        checks,
        "expected_boxes",
        not position_required or bool(expected_boxes),
        f"expected box count={len(expected_boxes)}",
    )
    missing_box_classes = _missing_expected_box_classes(expected_items, expected_boxes)
    _add(
        checks,
        "expected_box_coverage",
        not position_required or not missing_box_classes,
        "all expected item classes have at least one expected box"
        if not missing_box_classes
        else f"missing expected boxes for: {', '.join(missing_box_classes)}",
    )

    conf = _as_float(config.get("conf_thres"), default=0.25)
    iou = _as_float(config.get("iou_thres"), default=0.45)
    _add(checks, "conf_threshold_range", 0.0 < conf <= 1.0, f"conf_thres={conf}")
    _add(checks, "iou_threshold_range", 0.0 < iou <= 1.0, f"iou_thres={iou}")
    _add_position_tolerance_check(checks, position_cfg)
    _add_alignment_quality_gate_check(checks, position_cfg)
    _add_defect_coverage_checks(checks, config)

    _add_required_boolean_check(
        checks,
        "save_original",
        config,
        "save_original",
        default=True,
        message="raw image evidence should be saved",
    )
    _add_required_boolean_check(
        checks,
        "save_annotated",
        config,
        "save_annotated",
        default=True,
        message="annotated image evidence should be saved",
    )
    _add_required_boolean_check(
        checks,
        "save_crops",
        config,
        "save_crops",
        default=True,
        message="NG crop evidence should be saved",
    )
    _add(
        checks,
        "output_dir",
        bool(str(config.get("output_dir") or "").strip()),
        f"output_dir={config.get('output_dir') or '-'}",
    )
    _add_required_boolean_check(
        checks,
        "fail_on_unexpected",
        config,
        "fail_on_unexpected",
        default=True,
        message="unexpected classes should fail in production",
    )
    color_model_path = _add_color_readiness_checks(checks, effective_path.parent, config)
    if source_paths is not None and color_model_path is not None:
        source_paths.add(color_model_path.expanduser().resolve(strict=False))

    missing_slot = position_cfg.get("missing_slot_check") if isinstance(position_cfg, dict) else {}
    if isinstance(missing_slot, dict):
        missing_slot_enabled = missing_slot.get("enabled", False)
        _add(
            checks,
            "missing_slot_check",
            missing_slot_enabled is True,
            "recommended for missing-item false fail reduction"
            if isinstance(missing_slot_enabled, bool)
            else "missing_slot_check.enabled must be a YAML boolean",
            warn_only=True,
        )
    else:
        _add(checks, "missing_slot_check", False, "missing_slot_check is not configured", warn_only=True)

    return checks


def has_blocking_failures(checks: list[ReadinessCheck]) -> bool:
    """Return True when any check has FAIL status."""
    return any(check.status == "FAIL" for check in checks)


def write_report(
    checks: list[ReadinessCheck],
    output_json: str | Path | None = None,
    *,
    protected_sources: Iterable[str | Path] = (),
) -> None:
    """Optionally write readiness checks to collision-safe atomic JSON."""
    if output_json is None:
        return
    candidate = Path(output_json).expanduser()
    if candidate.is_symlink():
        raise ValueError("readiness report destination cannot be a symbolic link")
    destination = candidate.resolve(strict=False)
    sources = {Path(source).expanduser().resolve(strict=False) for source in protected_sources}
    if destination in sources:
        raise ValueError("readiness report destination cannot overwrite source evidence")
    if destination.suffix.lower() != ".json":
        raise ValueError("readiness report destination must use a .json suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                [asdict(check) for check in checks],
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _load_effective_config(path: Path, product: str | None, area: str | None) -> tuple[dict[str, Any], Path]:
    """Merge the selected model config when a global config is supplied."""
    base = _load_yaml(path)
    product_name = product or str(base.get("current_product") or "")
    area_name = area or str(base.get("current_area") or "")
    if not product_name or not area_name or "models" in path.parts:
        return base, path

    relative = Path("models") / product_name / area_name / "yolo" / "config.yaml"
    candidates: list[Path] = []
    if (path.parent / "models").is_dir():
        candidates.append(path.parent / relative)
    if path.parent.resolve() == Path.cwd().resolve():
        candidates.append(Path.cwd() / relative)
    if not candidates:
        return base, path
    model_path = next((candidate for candidate in candidates if candidate.exists()), None)
    if model_path is None:
        return base, candidates[0]
    model = _load_yaml(model_path)
    effective = dict(base)
    effective.update(model)
    if isinstance(base.get("steps"), dict) and isinstance(model.get("steps"), dict):
        effective["steps"] = {**base["steps"], **model["steps"]}
    return effective, model_path


def _position_required(config: dict[str, Any], position_cfg: dict[str, Any]) -> bool:
    """Return whether declared inspection scope requires positional checks."""
    coverage = config.get("defect_coverage") or {}
    covered = coverage.get("covered", []) if isinstance(coverage, dict) else []
    required_defects = {"missing_component", "position_shift", "wrong_position"}
    if covered:
        return bool(required_defects.intersection(_normalize_string_list(covered)))
    enabled = position_cfg.get("enabled", False)
    return enabled if isinstance(enabled, bool) else True


def _resolve_existing_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        (base_dir / path).resolve(),
        (Path.cwd() / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _expected_items(config: dict[str, Any], product: str, area: str) -> list[str]:
    values = config.get("expected_items", {})
    if not isinstance(values, dict):
        return []
    items = values.get(product, {}).get(area, []) if product and area else []
    return [str(item).strip() for item in items if str(item).strip()] if isinstance(items, list) else []


def _position_config(config: dict[str, Any], product: str, area: str) -> dict[str, Any]:
    values = config.get("position_config", {})
    if not isinstance(values, dict) or not product or not area:
        return {}
    area_cfg = values.get(product, {}).get(area, {})
    return area_cfg if isinstance(area_cfg, dict) else {}


def _missing_expected_box_classes(expected_items: list[str], expected_boxes: dict[str, Any]) -> list[str]:
    box_bases = {_base_class_name(str(key)) for key in expected_boxes}
    return sorted({item for item in expected_items if item not in box_bases})


def _base_class_name(key: str) -> str:
    idx = key.rfind("#")
    if idx > 0 and key[idx + 1 :].isdigit():
        return key[:idx]
    return key


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _add_position_tolerance_check(
    checks: list[ReadinessCheck],
    position_cfg: dict[str, Any],
) -> None:
    mode = str(position_cfg.get("mode", "center")).lower()
    tolerance = _as_float(position_cfg.get("tolerance"), default=0.0)
    tolerance_unit = str(position_cfg.get("tolerance_unit", "percent")).lower()
    if mode == "iou":
        min_iou = tolerance / 100.0 if tolerance > 1.0 else tolerance
        _add(
            checks,
            "position_iou_tolerance",
            min_iou >= 0.3,
            f"effective minimum IoU={min_iou:.4f}; verify this is intentional for production",
            warn_only=True,
        )
        return
    if tolerance_unit == "percent":
        _add(
            checks,
            "position_tolerance_percent",
            0.0 < tolerance <= 5.0,
            f"position tolerance={tolerance}% of image size; verify fixture variation supports this",
            warn_only=True,
        )
    else:
        _add(
            checks,
            "position_tolerance_pixel",
            tolerance > 0,
            f"position tolerance={tolerance}px",
            warn_only=True,
        )


def _add_color_readiness_checks(
    checks: list[ReadinessCheck],
    base_dir: Path,
    config: dict[str, Any],
) -> Path | None:
    enabled = config.get("enable_color_check", False)
    if not isinstance(enabled, bool):
        _add(
            checks,
            "color_check_enabled",
            False,
            "enable_color_check must be a YAML boolean",
        )
        return None
    _add(
        checks,
        "color_check_enabled",
        True,
        f"enabled={str(enabled).lower()}",
    )
    if not enabled:
        return None

    color_model = str(config.get("color_model_path") or "").strip()
    color_model_path = _resolve_existing_path(base_dir, color_model) if color_model else None
    _add(
        checks,
        "color_model_configured",
        bool(color_model),
        "color_model_path is required when enable_color_check is true",
    )
    _add(
        checks,
        "color_model_exists",
        bool(color_model_path and color_model_path.exists()),
        f"color_model_path={color_model_path or '-'}",
    )
    _add(
        checks,
        "color_fail_closed",
        config.get("color_fail_closed", True) is True,
        "color checker failures should block production inspections",
    )
    return color_model_path


def _add_alignment_quality_gate_check(
    checks: list[ReadinessCheck],
    position_cfg: dict[str, Any],
) -> None:
    alignment = position_cfg.get("alignment") if isinstance(position_cfg, dict) else {}
    alignment = alignment if isinstance(alignment, dict) else {}
    gate = alignment.get("quality_gate") if isinstance(alignment, dict) else {}
    gate = gate if isinstance(gate, dict) else {}
    raw_enabled = gate.get("enabled", False)
    enabled = raw_enabled if isinstance(raw_enabled, bool) else False
    _add(
        checks,
        "alignment_quality_gate",
        enabled,
        "recommended to fail when board alignment sources or shift exceed limits"
        if isinstance(raw_enabled, bool)
        else "alignment quality_gate.enabled must be a YAML boolean",
        warn_only=True,
    )
    if not enabled:
        return

    has_limit = any(key in gate for key in ("max_abs_dx_px", "max_abs_dy_px", "max_shift_px"))
    _add(
        checks,
        "alignment_shift_limits",
        has_limit,
        "quality_gate should define max_abs_dx_px/max_abs_dy_px or max_shift_px",
    )


def _add_defect_coverage_checks(
    checks: list[ReadinessCheck],
    config: dict[str, Any],
) -> None:
    coverage = config.get("defect_coverage")
    if not isinstance(coverage, dict):
        _add(
            checks,
            "defect_coverage_declared",
            False,
            "declare covered/not_covered defect types so production scope is explicit",
            warn_only=True,
        )
        return

    covered = _normalize_string_list(coverage.get("covered"))
    not_covered = _normalize_string_list(coverage.get("not_covered"))
    _add(
        checks,
        "defect_coverage_declared",
        bool(covered),
        f"covered defect types={', '.join(covered) if covered else '-'}",
    )
    _add(
        checks,
        "defect_coverage_limitations",
        not bool(not_covered),
        "not covered: " + ", ".join(not_covered) if not_covered else "no uncovered defect types declared",
        warn_only=True,
    )


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _add_required_boolean_check(
    checks: list[ReadinessCheck],
    name: str,
    config: dict[str, Any],
    key: str,
    *,
    default: bool,
    message: str,
) -> None:
    """Require an enabled YAML boolean without truthy-string coercion."""
    value = config.get(key, default)
    _add(
        checks,
        name,
        value is True,
        message if isinstance(value, bool) else f"{key} must be a YAML boolean",
    )


def _add(
    checks: list[ReadinessCheck],
    name: str,
    passed: bool,
    message: str,
    *,
    warn_only: bool = False,
) -> None:
    status = "PASS" if passed else ("WARN" if warn_only else "FAIL")
    checks.append(ReadinessCheck(name=name, status=status, message=message))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Inference config YAML path")
    parser.add_argument("--product", default=None, help="Product override")
    parser.add_argument("--area", default=None, help="Area override")
    parser.add_argument("--output-json", default=None, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    source_paths: set[Path] = set()
    checks = run_readiness_checks(
        args.config,
        product=args.product,
        area=args.area,
        source_paths=source_paths,
    )
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.message}")
    try:
        write_report(
            checks,
            args.output_json,
            protected_sources=source_paths,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"[FAIL] readiness_report: {exc}")
        return 1
    return 1 if has_blocking_failures(checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
