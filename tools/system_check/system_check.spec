# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone preflight checker.

Build from the repository root:

    pyinstaller --clean --noconfirm tools/system_check/system_check.spec

The result is ``dist/system_check/system_check.exe``, which runs on a Windows
machine with no Python installed.

Torch, Ultralytics, PyQt5, anomalib and the rest of the application stack are
excluded on purpose. Bundling torch alone would add over two gigabytes to a
tool whose job is to be copied onto a new machine before anything else is
installed, and the checker does not need them:

* The benchmark drives ``onnxruntime`` directly, which is what the production
  model bundles deploy (``.onnx`` weights, ``device: cpu``).
* Package versions of a *packaged* application are read from its ``_internal``
  metadata, not by importing them here.
* ``torch.cuda`` and the Hikrobot ``MvImport`` bindings are imported lazily and
  report SKIP / UNKNOWN when absent, naming the application command that
  answers the same question.

Run the checker from the application's own Python environment instead when you
want the Ultralytics benchmark backend or a CUDA answer from torch.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(SPECPATH, "..", ".."))

# A conda interpreter keeps the DLLs its own stdlib extensions link against —
# libffi for ``_ctypes``, libexpat for ``pyexpat``, libssl for ``_ssl`` — in
# ``<prefix>/Library/bin`` rather than beside the .pyd files. PyInstaller
# resolves native dependencies by searching PATH, and that directory is only on
# PATH inside an activated environment. Building with an unactivated
# interpreter therefore produces an executable that dies at startup with
# "DLL load failed while importing _ctypes", before any project code runs.
#
# Putting those directories on PATH for the build process makes the build
# correct whether or not the environment was activated, and lets PyInstaller
# discover the dependencies itself instead of relying on a hand-maintained
# list that would rot the next time a runtime is added.
_search_dirs = []
for _prefix in (sys.prefix, sys.base_prefix):
    for _relative in ("Library/bin", "Library/mingw-w64/bin", "Library/usr/bin", "DLLs"):
        _candidate = os.path.join(_prefix, *_relative.split("/"))
        if os.path.isdir(_candidate) and _candidate not in _search_dirs:
            _search_dirs.append(_candidate)
if _search_dirs:
    os.environ["PATH"] = os.pathsep.join([*_search_dirs, os.environ.get("PATH", "")])

a = Analysis(
    [os.path.join(SPECPATH, "__main__.py")],
    pathex=[project_root],
    binaries=[],
    datas=[],
    hiddenimports=[
        "pyexpat",
        "onnxruntime",
        "onnxruntime.capi.onnxruntime_pybind11_state",
        "cv2",
        "numpy",
        "yaml",
        "psutil",
        "tools.system_check.checks.camera_check",
        "tools.system_check.checks.cpu_check",
        "tools.system_check.checks.dependency_check",
        "tools.system_check.checks.disk_check",
        "tools.system_check.checks.gpu_check",
        "tools.system_check.checks.memory_check",
        "tools.system_check.checks.network_check",
        "tools.system_check.checks.os_check",
        "tools.system_check.checks.runtime_check",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "ultralytics",
        "anomalib",
        "lightning",
        "pytorch_lightning",
        "timm",
        "kornia",
        "FrEIA",
        "open_clip",
        "PyQt5",
        "matplotlib",
        "pandas",
        "scipy",
        "sklearn",
        "skimage",
        "imgaug",
        "PIL",
        "tkinter",
        "_tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="system_check",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="system_check",
)
