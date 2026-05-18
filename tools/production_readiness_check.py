from __future__ import annotations

"""Validate whether an inference config is ready for controlled production use."""

import argparse
import json
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
) -> list[ReadinessCheck]:
    """Run production readiness checks against a YAML config.

    Args:
        config_path: Global or model-specific inference config.
        product: Product override. Falls back to ``current_product``.
        area: Area override. Falls back to ``current_area``.

    Returns:
        Ordered readiness check results.
    """
    path = Path(config_path)
    config = _load_yaml(path)
    product_name = product or str(config.get("current_product") or "")
    area_name = area or str(config.get("current_area") or "")
    checks: list[ReadinessCheck] = []

    _add(checks, "config_exists", path.exists(), f"config={path}")
    _add(checks, "product_area", bool(product_name and area_name), f"product={product_name or '-'}, area={area_name or '-'}")

    weights = str(config.get("weights") or "")
    weights_path = _resolve_existing_path(path.parent, weights) if weights else None
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
    _add(
        checks,
        "position_check_enabled",
        bool(position_cfg.get("enabled")),
        "position_config must be enabled for production PCBA missing/shift checks",
    )
    expected_boxes = position_cfg.get("expected_boxes") if isinstance(position_cfg, dict) else {}
    expected_boxes = expected_boxes if isinstance(expected_boxes, dict) else {}
    _add(checks, "expected_boxes", bool(expected_boxes), f"expected box count={len(expected_boxes)}")
    missing_box_classes = _missing_expected_box_classes(expected_items, expected_boxes)
    _add(
        checks,
        "expected_box_coverage",
        not missing_box_classes,
        "all expected item classes have at least one expected box"
        if not missing_box_classes
        else f"missing expected boxes for: {', '.join(missing_box_classes)}",
    )

    conf = _as_float(config.get("conf_thres"), default=0.25)
    iou = _as_float(config.get("iou_thres"), default=0.45)
    _add(checks, "conf_threshold_range", 0.0 < conf <= 1.0, f"conf_thres={conf}")
    _add(checks, "iou_threshold_range", 0.0 < iou <= 1.0, f"iou_thres={iou}")
    _add_position_tolerance_check(checks, position_cfg)

    _add(checks, "save_original", bool(config.get("save_original", True)), "raw image evidence should be saved")
    _add(checks, "save_annotated", bool(config.get("save_annotated", True)), "annotated image evidence should be saved")
    _add(checks, "save_crops", bool(config.get("save_crops", True)), "NG crop evidence should be saved")
    _add(checks, "output_dir", bool(str(config.get("output_dir") or "").strip()), f"output_dir={config.get('output_dir') or '-'}")
    _add(checks, "fail_on_unexpected", bool(config.get("fail_on_unexpected", True)), "unexpected classes should fail in production")
    _add_color_readiness_checks(checks, path.parent, config)

    missing_slot = position_cfg.get("missing_slot_check") if isinstance(position_cfg, dict) else {}
    if isinstance(missing_slot, dict):
        _add(
            checks,
            "missing_slot_check",
            bool(missing_slot.get("enabled", False)),
            "recommended for missing-item false fail reduction",
            warn_only=True,
        )
    else:
        _add(checks, "missing_slot_check", False, "missing_slot_check is not configured", warn_only=True)

    return checks


def has_blocking_failures(checks: list[ReadinessCheck]) -> bool:
    """Return True when any check has FAIL status."""
    return any(check.status == "FAIL" for check in checks)


def write_report(checks: list[ReadinessCheck], output_json: str | Path | None = None) -> None:
    """Optionally write readiness checks to JSON."""
    if output_json is None:
        return
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(check) for check in checks], handle, ensure_ascii=False, indent=2)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


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
) -> None:
    if not bool(config.get("enable_color_check", False)):
        return

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
        bool(config.get("color_fail_closed", True)),
        "color checker failures should block production inspections",
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
    checks = run_readiness_checks(args.config, product=args.product, area=args.area)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.message}")
    write_report(checks, args.output_json)
    return 1 if has_blocking_failures(checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
