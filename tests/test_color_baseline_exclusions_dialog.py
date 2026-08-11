from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

import app.gui.color_baseline_exclusions_dialog as dialog_module
from app.gui.color_baseline_exclusions_dialog import (
    ColorBaselineExclusionsDialog,
    _trusted_image_path,
)
from core.services.color_baseline_evidence import (
    ColorBaselineEvidenceExclusion,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


def _manifest(source_root: Path) -> Path:
    source_root.mkdir(parents=True, exist_ok=True)
    manifest = source_root / "feedback.csv"
    manifest.write_text("sample_id\n", encoding="utf-8")
    return manifest


def _valid_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = QImage(4, 4, QImage.Format_RGB32)
    image.fill(0xFF336699)
    assert image.save(str(path), "PNG")
    return path


def _exclusion(manifest: Path, image_path: Path | str) -> ColorBaselineEvidenceExclusion:
    return ColorBaselineEvidenceExclusion(
        sample_id="case-1",
        source_kind="color_review",
        source_manifest=str(manifest),
        image_path=str(image_path),
        image_sha256="a" * 64,
        reason_code="IMAGE_SHA256_MISMATCH",
        reason="影像內容與 manifest 不一致。",
    )


def test_dialog_opens_decodable_image_inside_manifest_source_root(
    tmp_path: Path,
    qapp,
    monkeypatch,
) -> None:
    manifest = _manifest(tmp_path / "color_review")
    image_path = _valid_png(manifest.parent / "images" / "case-1.png")
    opened_urls = []
    monkeypatch.setattr(
        dialog_module,
        "QDesktopServices",
        SimpleNamespace(
            openUrl=lambda url: opened_urls.append(url) is None,
        ),
    )

    dialog = ColorBaselineExclusionsDialog((_exclusion(manifest, image_path),))
    try:
        qapp.processEvents()
        assert dialog.open_image_button.isEnabled()
        dialog._open_selected_image()
        assert len(opened_urls) == 1
        assert Path(opened_urls[0].toLocalFile()) == image_path.resolve()
    finally:
        dialog.close()


@pytest.mark.parametrize("filename", ("payload.exe", "notes.txt", "photo.png"))
def test_dialog_rejects_existing_file_outside_manifest_source_root(
    tmp_path: Path,
    qapp,
    monkeypatch,
    filename: str,
) -> None:
    manifest = _manifest(tmp_path / "color_review")
    outside_path = tmp_path / "outside" / filename
    outside_path.parent.mkdir()
    if outside_path.suffix == ".png":
        _valid_png(outside_path)
    else:
        outside_path.write_bytes(b"untrusted content")
    exclusion = _exclusion(manifest, outside_path)
    opened_urls = []
    monkeypatch.setattr(
        dialog_module,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened_urls.append(url) is None),
    )
    monkeypatch.setattr(
        dialog_module,
        "QMessageBox",
        SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    assert _trusted_image_path(exclusion) is None
    dialog = ColorBaselineExclusionsDialog((exclusion,))
    try:
        qapp.processEvents()
        assert not dialog.open_image_button.isEnabled()
        dialog._open_selected_image()
        assert opened_urls == []
    finally:
        dialog.close()


def test_dialog_rejects_decodable_content_with_non_image_extension(
    tmp_path: Path,
    qapp,
    monkeypatch,
) -> None:
    manifest = _manifest(tmp_path / "color_review")
    executable_path = manifest.parent / "images" / "payload.exe"
    _valid_png(executable_path)
    exclusion = _exclusion(manifest, executable_path)
    opened_urls = []
    monkeypatch.setattr(
        dialog_module,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened_urls.append(url) is None),
    )
    monkeypatch.setattr(
        dialog_module,
        "QMessageBox",
        SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    assert _trusted_image_path(exclusion) is None
    dialog = ColorBaselineExclusionsDialog((exclusion,))
    try:
        qapp.processEvents()
        assert not dialog.open_image_button.isEnabled()
        dialog._open_selected_image()
        assert opened_urls == []
    finally:
        dialog.close()


def test_dialog_rejects_image_suffix_when_content_cannot_decode(
    tmp_path: Path,
    qapp,
    monkeypatch,
) -> None:
    manifest = _manifest(tmp_path / "color_review")
    disguised_image = manifest.parent / "images" / "document.png"
    disguised_image.parent.mkdir()
    disguised_image.write_bytes(b"not an image")
    exclusion = _exclusion(manifest, disguised_image)
    opened_urls = []
    monkeypatch.setattr(
        dialog_module,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened_urls.append(url) is None),
    )
    monkeypatch.setattr(
        dialog_module,
        "QMessageBox",
        SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    assert _trusted_image_path(exclusion) is None
    dialog = ColorBaselineExclusionsDialog((exclusion,))
    try:
        qapp.processEvents()
        assert not dialog.open_image_button.isEnabled()
        dialog._open_selected_image()
        assert opened_urls == []
    finally:
        dialog.close()
