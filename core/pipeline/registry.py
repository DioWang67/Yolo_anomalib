from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from core.pipeline.steps import Step, ColorCheckStep, SaveResultsStep, PositionCheckStep
from core.config import DetectionConfig
from core.services.color_checker import ColorCheckerService
from core.services.result_sink import ExcelImageResultSink


@dataclass(frozen=True)
class PipelineEnv:
    color_service: ColorCheckerService
    result_sink: ExcelImageResultSink
    logger: any
    product: str
    area: str
    config: DetectionConfig


StepFactory = Callable[[PipelineEnv, Dict], Optional[Step]]


_REGISTRY: Dict[str, StepFactory] = {}


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


def available_steps() -> List[str]:
    """Return the list of registered step names."""
    return sorted(_REGISTRY.keys())


def create_step(
    name: str, env: PipelineEnv, options: Optional[Dict] = None
) -> Optional[Step]:
    """Instantiate a registered step.

    Returns None when the factory decides to skip (e.g. disabled feature)."""
    key = name.strip().lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise KeyError(f"Unknown pipeline step: {name}")
    return factory(env, options or {})


def build_pipeline(
    step_names: Iterable[str], env: PipelineEnv, step_options: Dict[str, Dict]
) -> List[Step]:
    """Create step instances for the provided names in order."""
    steps: List[Step] = []
    seen_save = False
    for raw_name in step_names:
        key = str(raw_name).strip().lower()
        try:
            step = create_step(key, env, step_options.get(key, {}))
        except KeyError:
            env.logger.warning(f"Unknown pipeline step: {raw_name}")
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


def default_pipeline(env: PipelineEnv) -> List[str]:
    """Return the default pipeline order given current config."""
    names: List[str] = []
    cfg = env.config
    if getattr(cfg, "enable_color_check", False) and getattr(
        cfg, "color_model_path", None
    ):
        names.append("color_check")
    names.append("save_results")
    return names


# ---------------------------------------------------------------------------
# Default step registrations


def _color_step_factory(env: PipelineEnv, options: Dict) -> Optional[Step]:
    cfg = env.config
    if not getattr(cfg, "enable_color_check", False):
        return None
    if not getattr(cfg, "color_model_path", None):
        env.logger.warning(
            "Color check enabled but color_model_path is missing; skipping step"
        )
        return None
    if not env.color_service.is_ready():
        env.logger.warning(
            "Color checker not ready; skipping color_check step")
        return None
    return ColorCheckStep(env.color_service, env.logger, options=options)


def _save_step_factory(env: PipelineEnv, options: Dict) -> Optional[Step]:
    return SaveResultsStep(env.result_sink, env.logger, options=options)


def _position_step_factory(env: PipelineEnv, options: Dict) -> Optional[Step]:
    return PositionCheckStep(
        env.logger, product=env.product, area=env.area, options=options
    )


register_step("color_check", _color_step_factory)
register_step("save_results", _save_step_factory)
register_step("position_check", _position_step_factory)
