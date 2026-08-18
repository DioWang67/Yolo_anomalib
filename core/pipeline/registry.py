from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from core.config import DetectionConfig
from core.pipeline.steps import (
    ColorCheckStep,
    CountCheckStep,
    CrossClassDuplicateFilterStep,
    PositionCheckStep,
    SaveResultsStep,
    SequenceCheckStep,
    Step,
)
from core.services.color_checker import ColorCheckerService
from core.services.result_sink import ExcelImageResultSink


@dataclass(frozen=True)
class PipelineEnv:
    color_service: ColorCheckerService
    result_sink: ExcelImageResultSink
    logger: logging.Logger | logging.LoggerAdapter
    product: str
    area: str
    config: DetectionConfig


StepFactory = Callable[[PipelineEnv, dict], Step | None]


_REGISTRY: dict[str, StepFactory] = {}


def register_step(name: str, factory: StepFactory) -> None:
    """Register a pipeline step factory by name (case-insensitive)."""
    key = name.strip().lower()
    if not key:
        raise ValueError("Step name must not be empty")
    _REGISTRY[key] = factory


def unregister_step(name: str) -> None:
    """Remove a previously registered step if it exists."""
    key = name.strip().lower()
    _REGISTRY.pop(key, None)


def available_steps() -> list[str]:
    """Return the list of registered step names."""
    return sorted(_REGISTRY.keys())


def create_step(
    name: str, env: PipelineEnv, options: dict | None = None
) -> Step | None:
    """Instantiate a registered step.

    Returns None when the factory decides to skip (e.g. disabled feature)."""
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise KeyError(f"Unknown pipeline step: {name}")
    return factory(env, options or {})


def build_pipeline(
    step_names: Iterable[str], env: PipelineEnv, step_options: dict[str, dict]
) -> list[Step]:
    """Create step instances for the provided names in order."""
    normalized_names = [
        str(raw_name).strip().lower() for raw_name in step_names
    ]
    validate_duplicate_filter_order(normalized_names)
    steps: list[Step] = []
    seen_save = False
    for key in normalized_names:
        try:
            step = create_step(key, env, step_options.get(key, {}))
        except KeyError:
            env.logger.warning(f"Unknown pipeline step: {key}")
            continue
        if step is None:
            env.logger.debug(f"Pipeline step '{key}' skipped by factory")
            continue
        if key == "save_results":
            seen_save = True
        steps.append(step)
    if not seen_save:
        extra = create_step("save_results", env,
                            step_options.get("save_results", {}))
        if extra is not None:
            env.logger.info(
                "save_results step not present in pipeline; appended by default"
            )
            steps.append(extra)
    return steps


def validate_duplicate_filter_order(step_names: Iterable[str]) -> None:
    """Reject pipeline orders that cannot safely run duplicate filtering."""

    normalized_names = [
        str(raw_name).strip().lower() for raw_name in step_names
    ]
    orchestration_steps = (
        "color_check",
        "position_check",
        "cross_class_duplicate_filter",
        "count_check",
        "sequence_check",
        "save_results",
    )
    counts = Counter(normalized_names)
    repeated_steps = [
        name for name in orchestration_steps if counts[name] > 1
    ]
    if repeated_steps:
        raise ValueError(
            "critical pipeline steps must not be repeated: "
            + ", ".join(repeated_steps)
        )
    duplicate_name = "cross_class_duplicate_filter"
    if duplicate_name not in normalized_names:
        return
    duplicate_index = normalized_names.index(duplicate_name)
    if "color_check" not in normalized_names:
        raise ValueError(
            "cross_class_duplicate_filter requires color_check in the pipeline"
        )
    if normalized_names.index("color_check") > duplicate_index:
        raise ValueError(
            "cross_class_duplicate_filter must run after color_check"
        )
    downstream = ("count_check", "sequence_check", "save_results")
    for name in downstream:
        if name in normalized_names and normalized_names.index(name) < duplicate_index:
            raise ValueError(
                f"cross_class_duplicate_filter must run before {name}"
            )


def default_pipeline(env: PipelineEnv) -> list[str]:
    """Return the default pipeline order given current config."""
    names: list[str] = []
    cfg = env.config
    if getattr(cfg, "enable_color_check", False) and getattr(
        cfg, "color_model_path", None
    ):
        names.append("color_check")
    names.append("save_results")
    return names


# ---------------------------------------------------------------------------
# Default step registrations


def _color_step_factory(env: PipelineEnv, options: dict) -> Step | None:
    cfg = env.config
    if not getattr(cfg, "enable_color_check", False):
        return None
    if not getattr(cfg, "color_model_path", None):
        env.logger.warning(
            "Color check enabled but color_model_path is missing; color_check will fail closed"
        )
    return ColorCheckStep(env.color_service, env.logger, options=options)


def _save_step_factory(env: PipelineEnv, options: dict) -> Step | None:
    return SaveResultsStep(env.result_sink, env.logger, options=options)


def _position_step_factory(env: PipelineEnv, options: dict) -> Step | None:
    return PositionCheckStep(
        env.logger, product=env.product, area=env.area, options=options
    )


def _count_step_factory(env: PipelineEnv, options: dict) -> Step | None:
    return CountCheckStep(
        env.logger, product=env.product, area=env.area, options=options
    )


def _cross_class_duplicate_filter_factory(
    env: PipelineEnv, options: dict
) -> Step | None:
    if not options.get("enabled", True):
        return None
    return CrossClassDuplicateFilterStep(env.logger, options=options)


def _sequence_step_factory(env: PipelineEnv, options: dict) -> Step | None:
    return SequenceCheckStep(
        env.logger, product=env.product, area=env.area, options=options
    )


register_step("color_check", _color_step_factory)
register_step("save_results", _save_step_factory)
register_step("position_check", _position_step_factory)
register_step("count_check", _count_step_factory)
register_step(
    "cross_class_duplicate_filter",
    _cross_class_duplicate_filter_factory,
)
register_step("sequence_check", _sequence_step_factory)
