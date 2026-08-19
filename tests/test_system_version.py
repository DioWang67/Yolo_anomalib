from __future__ import annotations

from pathlib import Path

from core import __version__
from core._version import __version__ as authoritative_version
from core.services.results import __version__ as results_version


def test_all_public_version_aliases_use_authoritative_system_version() -> None:
    assert __version__ == authoritative_version
    assert results_version == authoritative_version


def test_package_build_uses_authoritative_system_version() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_text = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dynamic = ["version"]' in config_text
    assert 'version = {attr = "core._version.__version__"}' in config_text
    assert f'version = "{authoritative_version}"' not in config_text
