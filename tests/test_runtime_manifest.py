from pathlib import Path

import pytest

from tools.packaging.write_runtime_manifest import build_runtime_manifest


def test_build_runtime_manifest_lists_runtime_files(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    runtime_dir = bundle_root / "_internal" / "Runtime"
    cl_protocol_dir = runtime_dir / "CLProtocol" / "Win64_x64"
    cl_protocol_dir.mkdir(parents=True)
    (runtime_dir / "MvCameraControl.dll").write_bytes(b"dll")
    (runtime_dir / "MvProducerGEV.cti").write_bytes(b"cti")
    (cl_protocol_dir / "GenCP_MD_VC120_v3_0_MV.dll").write_bytes(b"cl")

    lines = build_runtime_manifest(bundle_root)

    assert r"_internal\Runtime\MvCameraControl.dll | 3 bytes" in lines
    assert r"_internal\Runtime\MvProducerGEV.cti | 3 bytes" in lines
    assert (
        r"_internal\Runtime\CLProtocol\Win64_x64\GenCP_MD_VC120_v3_0_MV.dll | 2 bytes"
        in lines
    )
    assert r"GENICAM_GENTL64_PATH=_internal\Runtime" in lines
    assert r"MVCAM_GENICAM_CLPROTOCOL=_internal\Runtime\CLProtocol" in lines


def test_build_runtime_manifest_requires_runtime_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_runtime_manifest(tmp_path / "missing_bundle")
