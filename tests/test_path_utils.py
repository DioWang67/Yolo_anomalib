"""Tests for core.path_utils — project root resolution and path handling."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from core.path_utils import project_root, resolve_path


class TestProjectRoot:
    """Tests for project_root()."""

    def test_returns_existing_directory(self):
        root = project_root()
        assert root.is_dir()

    def test_returns_path_object(self):
        root = project_root()
        assert isinstance(root, Path)

    def test_contains_expected_subdirectories(self):
        root = project_root()
        assert (root / "core").is_dir()
        assert (root / "app").is_dir()

    def test_respects_yolo11_root_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YOLO11_ROOT", str(tmp_path))
        root = project_root()
        assert root == tmp_path.resolve()

    def test_frozen_mode(self, tmp_path, monkeypatch):
        """When sys.frozen is set, project_root should use sys.executable parent."""
        monkeypatch.delenv("YOLO11_ROOT", raising=False)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        fake_exe = tmp_path / "dist" / "app.exe"
        fake_exe.parent.mkdir(parents=True, exist_ok=True)
        fake_exe.touch()
        monkeypatch.setattr(sys, "executable", str(fake_exe))
        root = project_root()
        assert root == fake_exe.parent.resolve()


class TestResolvePath:
    """Tests for resolve_path()."""

    def test_none_returns_none(self):
        assert resolve_path(None) is None

    def test_empty_string_returns_none(self):
        assert resolve_path("") is None

    def test_absolute_path_returned_as_is(self, tmp_path):
        abs_path = str(tmp_path / "file.txt")
        result = resolve_path(abs_path)
        assert result == Path(abs_path)

    def test_relative_path_resolves_to_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YOLO11_ROOT", str(tmp_path))
        test_file = tmp_path / "data.csv"
        test_file.touch()
        result = resolve_path("data.csv")
        assert result is not None
        assert result.exists()
        assert result == test_file.resolve()

    def test_relative_path_fallback_when_not_exists(self, monkeypatch):
        monkeypatch.delenv("YOLO11_ROOT", raising=False)
        result = resolve_path("nonexistent_file_xyz.abc")
        # Should still return a Path (fallback), just won't exist
        assert result is not None
        assert isinstance(result, Path)
