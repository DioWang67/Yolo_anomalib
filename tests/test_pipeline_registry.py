import types

import pytest

from core.pipeline.registry import (
    PipelineEnv,
    available_steps,
    build_pipeline,
    default_pipeline,
    register_step,
    unregister_step,
)
from core.pipeline.steps import ColorCheckStep, SaveResultsStep, Step


class _StubLogger:
    def __init__(self):
        self.warning_messages = []
        self.info_messages = []
        self.debug_messages = []

    def warning(self, message):
        self.warning_messages.append(message)

    def info(self, message):
        self.info_messages.append(message)

    def debug(self, message):
        self.debug_messages.append(message)


class _StubColorService:
    def __init__(self, ready: bool = True):
        self._ready = ready

    def is_ready(self) -> bool:
        return self._ready


class _StubSink:
    def save(self, *args, **kwargs):
        return {}

    def flush(self):
        pass


def _make_env(enable_color: bool = True, color_ready: bool = True):
    logger = _StubLogger()
    color_service = _StubColorService(ready=color_ready)
    sink = _StubSink()
    config = types.SimpleNamespace(
        enable_color_check=enable_color,
        color_model_path="model.json" if enable_color else None,
    )
    return PipelineEnv(
        color_service=color_service,
        result_sink=sink,
        logger=logger,
        product="LED",
        area="A",
        config=config,
    )


def test_default_pipeline_includes_color_when_ready():
    env = _make_env(enable_color=True, color_ready=True)
    steps = default_pipeline(env)
    assert steps == ["color_check", "save_results"]
    built = build_pipeline(steps, env, {})
    assert len(built) == 2
    assert isinstance(built[0], ColorCheckStep)
    assert isinstance(built[1], SaveResultsStep)


def test_default_pipeline_skips_color_when_disabled():
    env = _make_env(enable_color=False)
    steps = default_pipeline(env)
    assert steps == ["save_results"]
    built = build_pipeline(steps, env, {})
    assert len(built) == 1
    assert isinstance(built[0], SaveResultsStep)


def test_custom_step_registration_and_autosave_append():
    class _DummyStep(Step):
        def run(self, ctx):  # pragma: no cover - simple no-op for registry testing
            ctx.dummy_step_ran = True

    def factory(env: PipelineEnv, options):
        return _DummyStep()

    register_step("dummy", factory)
    try:
        env = _make_env(enable_color=False)
        pipeline = build_pipeline(["dummy"], env, {})
        assert any(
            isinstance(step, SaveResultsStep) for step in pipeline
        ), "save_results should be appended"
        assert isinstance(pipeline[0], _DummyStep)
        assert pipeline[-1].__class__ is SaveResultsStep
    finally:
        unregister_step("dummy")


@pytest.mark.parametrize("name", ["color_check", "save_results", "position_check"])
def test_available_steps_contains_defaults(name):
    assert name in available_steps()
