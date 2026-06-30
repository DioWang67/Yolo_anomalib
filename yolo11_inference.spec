# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

# Resolve all bundled paths relative to this spec file so the build works no
# matter where the repo is checked out. PyInstaller injects ``SPECPATH`` as the
# directory containing the spec; previously these were hardcoded to
# ``D:\Git\robotlearning\yolo11_inference`` and broke on any other machine/path.
project_root = os.path.abspath(SPECPATH)

datas = [
    (os.path.join(project_root, 'Runtime'), 'Runtime'),
    (os.path.join(project_root, 'MvImport'), 'MvImport'),
]
# timm_cache holds the Patchcore backbone cache; bundle it only when present.
timm_cache_dir = os.path.join(project_root, 'timm_cache')
if os.path.isdir(timm_cache_dir):
    datas.append((timm_cache_dir, 'timm_cache'))

binaries = []
hiddenimports = ['torch', 'torch.nn.functional', 'torchvision', 'cv2', 'numpy', 'scipy', 'scipy.special._ufuncs', 'PIL', 'kornia', 'anomalib', 'lightning', 'ultralytics', 'onnx', 'onnxruntime', 'onnxruntime.capi.onnxruntime_pybind11_state', 'pandas', 'openpyxl', 'openpyxl.cell._writer', 'yaml', 'pydantic', 'tqdm', 'timm', 'einops', 'FrEIA', 'imgaug', 'serial', 'serial.tools', 'serial.tools.list_ports', 'PyQt5', 'PyQt5.sip', 'PyQt5.QtCore', 'PyQt5.QtGui', 'PyQt5.QtWidgets', 'pkg_resources', 'importlib.metadata', 'jsonargparse']
datas += collect_data_files('anomalib')
datas += collect_data_files('open_clip')
datas += collect_data_files('ultralytics')
datas += copy_metadata('torch')
datas += copy_metadata('ultralytics')
datas += copy_metadata('onnx')
datas += copy_metadata('onnxruntime')
datas += copy_metadata('anomalib')
datas += copy_metadata('lightning')
hiddenimports += collect_submodules('anomalib')
hiddenimports += collect_submodules('anomalib.models')
hiddenimports += collect_submodules('ultralytics')
hiddenimports += collect_submodules('lightning')
hiddenimports += collect_submodules('timm')
hiddenimports += collect_submodules('PyQt5')
tmp_ret = collect_all('kornia')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('jsonargparse')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    [os.path.join(project_root, 'GUI.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', '_tkinter', 'PIL._tkinter_finder'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='yolo11_inference',
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
    name='yolo11_inference',
)
