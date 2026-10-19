"""The station refuses Anomalib packages it cannot vouch for.

Packages come from Yolo11_auto_train (train/anomalib_compat.py). These tests
cover the stamp checks that run before a checkpoint is loaded, and pin the
probe image to reference values computed from the training project's
implementation: the two repositories never import each other, so a drift in
either copy of the formula would make every good model look incompatible.
"""

import hashlib

import numpy as np
import pytest

from core import anomalib_lightning_inference as runtime

# The probe block exactly as the training project writes it for image_size=256.
PROBE_256 = {
    "kind": "sincos_v2",
    "base": 0.45,
    "amplitude": 0.05,
    "row_frequency": 0.11,
    "column_frequency": 0.07,
    "channel_phase": 0.7,
    "detail_amplitude": 0.04,
    "detail_frequency": 1.3,
    "size": 391,
    "raw_pred_score": -0.0469,
    "abs_tolerance": 1e-4,
    "rel_tolerance": 1e-3,
}


def test_probe_matches_the_training_projects_reference_values():
    image = runtime._probe_image(PROBE_256).astype(np.float64)

    assert image.shape == (3, 391, 391)
    assert image.mean() == pytest.approx(0.449995586, abs=1e-8)
    assert image.std() == pytest.approx(0.031929384, abs=1e-8)
    for index, expected in {
        (0, 0, 0): 0.449999988,
        (1, 17, 200): 0.450729549,
        (2, 390, 390): 0.476070642,
        (0, 123, 45): 0.420663834,
        (2, 77, 301): 0.446561217,
    }.items():
        assert image[index] == pytest.approx(expected, abs=1e-6)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"stamped checkpoint bytes")
    return path


def _stamp(checkpoint, **overrides):
    stamp = {
        "artifact_schema": 2,
        "runtime": {"anomalib": "2.7.0", "torch": "2.13.0"},
        "class_path": "anomalib.models.EfficientAd",
        "image_size": 256,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "probe": dict(PROBE_256),
    }
    stamp.update(overrides)
    return stamp


@pytest.fixture(autouse=True)
def _installed_anomalib(monkeypatch):
    monkeypatch.setattr(runtime.metadata, "version", lambda name: "2.7.0")


def test_a_matching_stamp_is_accepted(checkpoint):
    runtime._check_stamp(_stamp(checkpoint), "anomalib.models.EfficientAd", str(checkpoint))


@pytest.mark.parametrize(
    ("stamp_for", "message"),
    [
        (lambda ckpt: None, "No compatibility stamp"),
        (lambda ckpt: _stamp(ckpt, artifact_schema=1), "artifact_schema"),
        (lambda ckpt: _stamp(ckpt, class_path="anomalib.models.Padim"), "Stamp is for"),
        (lambda ckpt: _stamp(ckpt, runtime={"anomalib": "1.2.0"}), "trained with anomalib 1.2.0"),
        (lambda ckpt: _stamp(ckpt, runtime={"anomalib": "2.6.2"}), "trained with anomalib 2.6.2"),
        (lambda ckpt: _stamp(ckpt, checkpoint_sha256="0" * 64), "is not the stamped file"),
    ],
    ids=["missing", "schema", "class", "anomalib-1x", "anomalib-minor", "sha256"],
)
def test_an_untrustworthy_stamp_is_refused(checkpoint, stamp_for, message):
    with pytest.raises(runtime.AnomalibModelIncompatible, match=message):
        runtime._check_stamp(stamp_for(checkpoint), "anomalib.models.EfficientAd", str(checkpoint))


def test_a_patch_release_of_anomalib_is_still_compatible(checkpoint, monkeypatch):
    monkeypatch.setattr(runtime.metadata, "version", lambda name: "2.7.3")
    runtime._check_stamp(_stamp(checkpoint), "anomalib.models.EfficientAd", str(checkpoint))
