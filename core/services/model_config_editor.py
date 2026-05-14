from __future__ import annotations

"""Small helper for editing per-model YAML configs safely."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ModelConfigEditError(ValueError):
    """Raised when a model config edit is invalid."""


@dataclass(frozen=True)
class ModelConfigEditResult:
    """Result returned after saving a model config.

    Args:
        config_path: Path to the updated YAML file.
        backup_path: Path to the backup file created before writing.
        values: Complete config mapping after applying edits.
    """

    config_path: Path
    backup_path: Path
    values: dict[str, Any]


SCALAR_FIELDS: tuple[str, ...] = (
    "weights",
    "device",
    "conf_thres",
    "iou_thres",
    "imgsz",
    "timeout",
    "output_dir",
    "enable_yolo",
    "enable_anomalib",
    "enable_color_check",
    "color_model_path",
    "color_checker_type",
    "color_score_threshold",
    "fail_on_unexpected",
    "save_original",
    "save_processed",
    "save_annotated",
    "save_crops",
    "save_fail_only",
)


def load_model_config(config_path: Path) -> dict[str, Any]:
    """Load a model config YAML file.

    Args:
        config_path: Path to ``models/<product>/<area>/<type>/config.yaml``.

    Returns:
        Mutable dictionary with the loaded YAML values.

    Raises:
        ModelConfigEditError: If the file is missing, unreadable, or not a mapping.
    """
    if not config_path.exists():
        raise ModelConfigEditError(f"找不到模型設定檔: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelConfigEditError(f"無法讀取模型設定檔: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ModelConfigEditError(f"模型設定檔 YAML 格式錯誤: {exc}") from exc

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ModelConfigEditError("模型設定檔必須是 YAML mapping")
    return dict(raw)


def save_model_config(
    config_path: Path,
    values: dict[str, Any],
) -> ModelConfigEditResult:
    """Write a complete model config mapping with a backup.

    Args:
        config_path: Existing YAML path.
        values: Complete mapping to write.

    Returns:
        Save result including backup path.

    Raises:
        ModelConfigEditError: If the payload is invalid or write fails.
    """
    if not isinstance(values, dict):
        raise ModelConfigEditError("模型設定內容必須是 mapping")
    if not config_path.exists():
        raise ModelConfigEditError(f"找不到模型設定檔: {config_path}")

    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    try:
        backup_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        config_path.write_text(
            yaml.safe_dump(
                values,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        raise ModelConfigEditError(f"寫入模型設定失敗: {exc}") from exc
    return ModelConfigEditResult(config_path=config_path, backup_path=backup_path, values=values)


def update_model_config(
    config_path: Path,
    changes: dict[str, Any],
    *,
    product: str,
    area: str,
) -> ModelConfigEditResult:
    """Apply validated common-field changes to a model config.

    Args:
        config_path: Existing model config path.
        changes: Flat mapping for supported scalar fields. ``expected_items``
            can be supplied as ``list[str]`` and is written under
            ``expected_items[product][area]``.
        product: Product key used for nested expected-items updates.
        area: Area key used for nested expected-items updates.

    Returns:
        Save result including the updated mapping.

    Raises:
        ModelConfigEditError: If values have invalid types/ranges.
    """
    data = load_model_config(config_path)
    sanitized = _sanitize_changes(changes)

    expected_items = sanitized.pop("expected_items", None)
    position_changes = {
        key: sanitized.pop(key)
        for key in list(sanitized.keys())
        if key.startswith("position_") or key in {"missing_slot_check_enabled"}
    }
    count_check_strict = sanitized.pop("count_check_strict", None)
    for key, value in sanitized.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value

    if expected_items is not None:
        nested = data.get("expected_items")
        if not isinstance(nested, dict):
            nested = {}
        product_map = nested.get(product)
        if not isinstance(product_map, dict):
            product_map = {}
        product_map[area] = expected_items
        nested[product] = product_map
        data["expected_items"] = nested

    if position_changes:
        _apply_position_changes(data, product, area, position_changes)

    if count_check_strict is not None:
        _apply_count_check_strict(data, count_check_strict)

    return save_model_config(config_path, data)


def _sanitize_changes(changes: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize editable config values."""
    output: dict[str, Any] = {}
    for key, value in changes.items():
        if key == "expected_items":
            output[key] = _normalize_expected_items(value)
            continue
        if key == "count_check_strict":
            output[key] = bool(value)
            continue
        if key in {
            "position_check_enabled",
            "position_alignment_enabled",
            "missing_slot_check_enabled",
        }:
            output[key] = bool(value)
            continue
        if key == "position_tolerance":
            number = float(value)
            if number < 0:
                raise ModelConfigEditError("position_tolerance 不可小於 0")
            output[key] = number
            continue
        if key == "position_tolerance_unit":
            unit = str(value).strip().lower()
            if unit not in {"percent", "pixel"}:
                raise ModelConfigEditError("position_tolerance_unit 只支援 percent 或 pixel")
            output[key] = unit
            continue
        if key == "position_mode":
            mode = str(value).strip().lower()
            if mode not in {"center", "region", "iou"}:
                raise ModelConfigEditError("position_mode 只支援 center、region 或 iou")
            output[key] = mode
            continue
        if key not in SCALAR_FIELDS:
            raise ModelConfigEditError(f"不支援編輯欄位: {key}")
        output[key] = _normalize_scalar(key, value)
    return output


def _normalize_scalar(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in {"weights", "device", "output_dir", "color_model_path", "color_checker_type"}:
        text = str(value).strip()
        return text or None
    if key in {
        "enable_yolo",
        "enable_anomalib",
        "enable_color_check",
        "fail_on_unexpected",
        "save_original",
        "save_processed",
        "save_annotated",
        "save_crops",
        "save_fail_only",
    }:
        return bool(value)
    if key in {"conf_thres", "iou_thres", "color_score_threshold"}:
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ModelConfigEditError(f"{key} 必須介於 0.0 到 1.0")
        return number
    if key == "timeout":
        number = int(value)
        if number < 0:
            raise ModelConfigEditError("timeout 不可小於 0")
        return number
    if key == "imgsz":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ModelConfigEditError("imgsz 必須包含寬與高")
        width = int(value[0])
        height = int(value[1])
        if width <= 0 or height <= 0:
            raise ModelConfigEditError("imgsz 寬高必須大於 0")
        return [width, height]
    return value


def _normalize_expected_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(",", "\n").splitlines()
    elif isinstance(value, (list, tuple)):
        raw_items = [str(item) for item in value]
    else:
        raise ModelConfigEditError("expected_items 必須是文字或清單")
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        item = str(raw).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
    return items


def _area_position_config(
    data: dict[str, Any],
    product: str,
    area: str,
) -> dict[str, Any]:
    position_config = data.get("position_config")
    if not isinstance(position_config, dict):
        position_config = {}
    product_config = position_config.get(product)
    if not isinstance(product_config, dict):
        product_config = {}
    area_config = product_config.get(area)
    if not isinstance(area_config, dict):
        area_config = {}
    product_config[area] = area_config
    position_config[product] = product_config
    data["position_config"] = position_config
    return area_config


def _apply_position_changes(
    data: dict[str, Any],
    product: str,
    area: str,
    changes: dict[str, Any],
) -> None:
    area_config = _area_position_config(data, product, area)
    mapping = {
        "position_check_enabled": "enabled",
        "position_tolerance": "tolerance",
        "position_tolerance_unit": "tolerance_unit",
        "position_mode": "mode",
    }
    for source_key, target_key in mapping.items():
        if source_key in changes:
            area_config[target_key] = changes[source_key]

    if "position_alignment_enabled" in changes:
        alignment = area_config.get("alignment")
        if not isinstance(alignment, dict):
            alignment = {}
        alignment["enabled"] = changes["position_alignment_enabled"]
        area_config["alignment"] = alignment

    if "missing_slot_check_enabled" in changes:
        missing_slot_check = area_config.get("missing_slot_check")
        if not isinstance(missing_slot_check, dict):
            missing_slot_check = {}
        missing_slot_check["enabled"] = changes["missing_slot_check_enabled"]
        area_config["missing_slot_check"] = missing_slot_check


def _apply_count_check_strict(data: dict[str, Any], strict: bool) -> None:
    steps = data.get("steps")
    if not isinstance(steps, dict):
        steps = {}
    count_check = steps.get("count_check")
    if not isinstance(count_check, dict):
        count_check = {}
    count_check["strict"] = bool(strict)
    steps["count_check"] = count_check
    data["steps"] = steps
