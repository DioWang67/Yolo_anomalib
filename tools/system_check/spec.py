"""Runtime requirements for yolo11_inference, derived from the repository.

Every value here is traceable to a file in this checkout, recorded in
``source``. Nothing is a guess: where the repository does not state a
requirement, the entry is marked :attr:`Confidence.UNKNOWN` and the checker
reports the measurement without passing judgement. Where a threshold is this
tool's own inference from repository facts, it is marked
:attr:`Confidence.SUGGESTED` and labelled as advisory in every report.

Keep this module free of imports from ``core``; it is the contract the checker
applies, not the application it checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tools.system_check.results import Confidence

# --------------------------------------------------------------------------
# Python runtime
# --------------------------------------------------------------------------

#: ``requires-python = ">=3.10,<3.12"`` (pyproject.toml).
PYTHON_MIN: tuple[int, int] = (3, 10)
PYTHON_MAX_EXCLUSIVE: tuple[int, int] = (3, 12)

#: The pair tools/check_runtime_environment.py already gates the launcher on.
SUPPORTED_PYTHON_MINORS: tuple[tuple[int, int], ...] = ((3, 10), (3, 11))

# --------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------

#: Distributions whose version tools/check_runtime_environment.py treats as a
#: hard gate. A mismatch here blocks start.bat today, so the checker reports it
#: as FAIL rather than drift.
PINNED_BLOCKING: dict[str, str] = {
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "onnxruntime": "1.23.2",
    "PyQt5": "5.15.11",
    "jsonargparse": "4.34.0",
}

#: Remaining runtime pins from requirements.txt. Drift is reported as WARNING:
#: the launcher does not block on these, but a mismatch changes behaviour
#: (numpy/opencv differences can move a borderline colour verdict).
PINNED_ADVISORY: dict[str, str] = {
    "numpy": "1.26.4",
    "opencv-python": "4.9.0.80",
    "ultralytics": "8.3.156",
    "pillow": "10.4.0",
    "PyYAML": "6.0.2",
    "pandas": "2.3.1",
    "openpyxl": "3.1.5",
    "scikit-learn": "1.5.2",
    "scipy": "1.13.1",
    "pyserial": "3.5",
}

#: Distributions only needed when ``enable_anomalib`` is true.
PINNED_ANOMALIB: dict[str, str] = {
    "anomalib": "1.2.0",
    "lightning": "2.4.0",
    "timm": "1.0.19",
    "kornia": "0.8.1",
    "einops": "0.8.1",
    "FrEIA": "0.2",
    "open-clip-torch": "3.1.0",
}

#: Imports tools/check_runtime_environment.py requires before the GUI starts.
REQUIRED_IMPORTS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "onnxruntime",
    "cv2",
    "PyQt5.QtCore",
    "jsonargparse",
    "yaml",
)

#: ONNX Runtime execution provider core/runtime_preflight.py demands before any
#: ``.onnx`` model is allowed to load.
REQUIRED_ORT_PROVIDER = "CPUExecutionProvider"

# --------------------------------------------------------------------------
# Native runtime
# --------------------------------------------------------------------------

#: Hikrobot MVS files GUI.py --check-hikrobot-runtime requires, relative to the
#: ``Runtime`` directory.
HIKROBOT_RUNTIME_FILES: tuple[str, ...] = (
    "MvCameraControl.dll",
    "MVGigEVisionSDK.dll",
    "MvUsb3vTL.dll",
    "MvProducerGEV.cti",
    "MvProducerU3V.cti",
    "GenApi_MD_VC120_v3_0_MV.dll",
    "GCBase_MD_VC120_v3_0_MV.dll",
    "CLProtocol/Win64_x64/GenCP_MD_VC120_v3_0_MV.dll",
)

#: Visual C++ 2015-2022 runtime DLLs. Not inferred from the redistributable's
#: name in core/runtime_preflight.py's error message: these are the exact
#: imports in the PE import tables of ``onnxruntime.dll`` and
#: ``onnxruntime_pybind11_state.pyd``. ONNX Runtime does not ship them, so they
#: must come from the installed redistributable.
VCRUNTIME_DLLS: tuple[str, ...] = (
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
)

#: Visual C++ 2013 runtime DLLs, imported by 14 binaries in ``Runtime`` —
#: including ``GenApi_MD_VC120_v3_0_MV.dll`` and ``GCBase_MD_VC120_v3_0_MV.dll``,
#: which the camera path needs. Verified from their PE import tables, not from
#: the ``_MD_VC120_`` filenames.
#:
#: The Hikrobot SDK ships its own copies inside ``Runtime``, and Windows
#: resolves a dependent DLL from the loading module's own directory, so this is
#: normally satisfied without any installed redistributable. The check must
#: therefore look in ``Runtime`` before concluding anything — searching only the
#: system path reports a missing runtime on a perfectly healthy station.
VC120_RUNTIME_DLLS: tuple[str, ...] = ("msvcr120.dll", "msvcp120.dll")

#: Older CRTs that Runtime binaries also import (MvISPControl, MvCameraPatch
#: and others need VC90; pthreadVC2 needs VC100). Also shipped inside
#: ``Runtime``, and listed here so the report can say so rather than leaving
#: them unmentioned.
LEGACY_RUNTIME_DLLS: tuple[str, ...] = (
    "msvcr90.dll",
    "msvcp90.dll",
    "msvcr100.dll",
)

# --------------------------------------------------------------------------
# Storage and memory
# --------------------------------------------------------------------------

#: core/config.py ``min_free_disk_mb`` default; the inference pipeline refuses
#: to write results below it.
MIN_FREE_DISK_MB = 1024

#: core/config.py ``image_queue_max_mb`` default — the hard ceiling on in-flight
#: image bytes held by the async pipeline.
IMAGE_QUEUE_MAX_MB = 256

#: core/config.py ``max_cache_size`` default — models held resident by the LRU.
MODEL_CACHE_SIZE = 3

# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------

#: core/async_pipeline.py starts acquisition, inference and storage workers;
#: the Qt GUI thread is the fourth concurrent consumer.
PIPELINE_THREADS = 3
SUGGESTED_LOGICAL_CORES = PIPELINE_THREADS + 1

# --------------------------------------------------------------------------
# Latency
# --------------------------------------------------------------------------

#: core/config.py ``timeout`` default, in seconds. core/fusion_inference.py
#: raises and fails the inspection when a backend exceeds it.
DEFAULT_INFERENCE_TIMEOUT_S = 2.0

#: The active Cable1/A model bundle tightens it to 1 second.
OBSERVED_MODEL_TIMEOUT_S = 1.0

#: Fraction of the configured timeout that P99 latency may consume before the
#: checker warns. Advisory: the repository states the hard timeout, not a
#: safety margin. This is about the *timeout*, not about line cycle time —
#: no cycle-time figure is derived anywhere in this tool.
SUGGESTED_P99_TIMEOUT_BUDGET = 0.5

# --------------------------------------------------------------------------
# Provenance table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """One runtime requirement and where it came from.

    Args:
        key: Stable identifier, matching the ``check_id`` that evaluates it.
        category: Grouping used by the report.
        statement: The requirement in words.
        confidence: Whether the repository states it, this tool inferred it, or
            it is undetermined.
        source: ``path:line`` inside this checkout, or ``""`` when unknown.
    """

    key: str
    category: str
    statement: str
    confidence: Confidence
    source: str = ""
    notes: str = ""


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        "os.platform",
        "OS",
        "Windows x64. The camera SDK is loaded with ctypes.WinDLL and the DLL "
        "search path is set with os.add_dll_directory; launchers are .bat.",
        Confidence.CONFIRMED,
        "MvImport/MvCameraControl_class.py:31-44, GUI.py:62-74",
    ),
    Requirement(
        "python.version",
        "Runtime",
        "CPython >=3.10,<3.12 (production station runs 3.10; CI compiles pins "
        "on 3.11).",
        Confidence.CONFIRMED,
        "pyproject.toml:18, tools/check_runtime_environment.py:13",
    ),
    Requirement(
        "deps.blocking",
        "Runtime",
        "torch 2.4.1, torchvision 0.19.1, onnxruntime 1.23.2, PyQt5 5.15.11, "
        "jsonargparse 4.34.0 at exactly these versions.",
        Confidence.CONFIRMED,
        "tools/check_runtime_environment.py:14-20",
    ),
    Requirement(
        "runtime.onnx_provider",
        "Runtime",
        "ONNX Runtime must import and expose CPUExecutionProvider before any "
        ".onnx model loads.",
        Confidence.CONFIRMED,
        "core/runtime_preflight.py:103-145",
    ),
    Requirement(
        "runtime.vcredist",
        "Runtime",
        "Microsoft Visual C++ Redistributable 2015-2022 x64. onnxruntime.dll "
        "and onnxruntime_pybind11_state.pyd import MSVCP140, MSVCP140_1, "
        "VCRUNTIME140 and VCRUNTIME140_1, and ONNX Runtime does not ship them.",
        Confidence.CONFIRMED,
        "PE import tables of onnxruntime.dll, core/runtime_preflight.py:131-133",
    ),
    Requirement(
        "runtime.vc120",
        "Runtime",
        "Visual C++ 2013 (and 2008/2010) CRTs, imported by 14 binaries in "
        "Runtime including GenApi and GCBase. Satisfied by the copies the "
        "Hikrobot SDK ships inside Runtime, so no separate redistributable is "
        "normally required.",
        Confidence.CONFIRMED,
        "PE import tables of Runtime/*_MD_VC120_*.dll; Runtime/msvcr120.dll",
        notes="Verified from import tables, not from the _MD_VC120_ filenames.",
    ),
    Requirement(
        "runtime.hikrobot",
        "Hardware",
        "Hikrobot MVS runtime: MvCameraControl.dll plus GigE/USB3 transport "
        "layers and GenICam CLProtocol, shipped in Runtime/.",
        Confidence.CONFIRMED,
        "GUI.py:35-44",
    ),
    Requirement(
        "disk.free",
        "Storage",
        f"At least {MIN_FREE_DISK_MB} MB free on the result volume; below it "
        "the pipeline stops writing inspection evidence.",
        Confidence.CONFIRMED,
        "core/config.py:199, core/config_schema.py:157",
    ),
    Requirement(
        "benchmark.latency",
        "Performance (enforced timeout)",
        f"One inference must complete within the configured timeout "
        f"({OBSERVED_MODEL_TIMEOUT_S:g} s for the active Cable1/A bundle, "
        f"{DEFAULT_INFERENCE_TIMEOUT_S:g} s by default) or the inspection is "
        "failed. This is a software constraint the application enforces, and "
        "is judged PASS/WARNING/FAIL.",
        Confidence.CONFIRMED,
        "core/config.py:145, core/fusion_inference.py:95-175",
        notes="A different concept from line cycle time; see throughput.fps.",
    ),
    Requirement(
        "memory.available",
        "Memory",
        f"No total-RAM figure is stated. The pipeline bounds in-flight images "
        f"at {IMAGE_QUEUE_MAX_MB} MB and keeps {MODEL_CACHE_SIZE} models "
        "resident; the checker judges RAM against its own measured peak "
        "instead of an invented minimum.",
        Confidence.SUGGESTED,
        "core/config.py:197-199",
        notes="Hard minimum is UNKNOWN.",
    ),
    Requirement(
        "cpu.cores",
        "CPU",
        f"No core count is stated. The async pipeline runs {PIPELINE_THREADS} "
        f"worker threads alongside the Qt GUI thread, so "
        f"{SUGGESTED_LOGICAL_CORES} logical cores is this tool's advisory "
        "floor.",
        Confidence.SUGGESTED,
        "core/async_pipeline.py:175-197",
        notes="Hard minimum is UNKNOWN.",
    ),
    Requirement(
        "gpu.present",
        "GPU",
        "No GPU is required. requirements.txt pins the CPU torch wheel and the "
        "CPU onnxruntime build, and the active model bundle pins device: cpu.",
        Confidence.CONFIRMED,
        "requirements.txt:158,279, models/Cable1/A/yolo/config.yaml:2",
        notes="CUDA support exists in code (core/yolo_runtime.py) but is "
        "unreachable with the pinned wheels.",
    ),
    Requirement(
        "camera.device",
        "Hardware",
        "A Hikrobot GigE or USB3 Vision camera. Image-file inference still "
        "works without one; live inspection does not.",
        Confidence.CONFIRMED,
        "GUI.py:134-141, MvImport/MvCameraControl_class.py:58-78",
    ),
    Requirement(
        "serial.light",
        "Hardware",
        "Optional serial LED controller at 115200 baud on a COM port.",
        Confidence.CONFIRMED,
        "core/services/light_controller.py:30",
    ),
    Requirement(
        "network.sync",
        "Network",
        "Outbound HTTPS to the inspection sync endpoint, only when "
        "inspection_sync_enabled is true. It ships disabled.",
        Confidence.CONFIRMED,
        "config.example.yaml:80-87",
    ),
    Requirement(
        "throughput.fps",
        "Performance (line cycle time)",
        "No FPS, cycle-time, takt-time or throughput requirement is stated "
        "anywhere in the repository. Throughput is measured and reported "
        "without a verdict; it is a property of the production line, not of "
        "the software, and nobody has written it down.",
        Confidence.UNKNOWN,
        "",
        notes="Distinct from the enforced inference timeout (benchmark.latency). "
        "auto_trigger.inspection_cooldown_ms (500) bounds how often a trigger "
        "may fire; it is not a throughput target.",
    ),
    Requirement(
        "cpu.instruction_set",
        "CPU",
        "No instruction-set requirement is stated. ONNX Runtime and torch x64 "
        "wheels select AVX2/AVX-512 kernels at runtime, so absence changes "
        "speed rather than correctness.",
        Confidence.UNKNOWN,
        "",
    ),
    Requirement(
        "memory.minimum",
        "Memory",
        "Minimum total RAM: UNKNOWN.",
        Confidence.UNKNOWN,
        "",
    ),
    Requirement(
        "gpu.vram",
        "GPU",
        "Minimum VRAM: not applicable while the pinned wheels are CPU-only.",
        Confidence.UNKNOWN,
        "",
    ),
)


@dataclass(frozen=True)
class ModelTarget:
    """A model bundle the benchmark can exercise.

    Args:
        product: Product directory name under ``models/``.
        area: Area directory name.
        weights: Absolute path to the weights artifact.
        imgsz: ``(height, width)`` the production config letterboxes to.
        conf_thres: Confidence threshold from the bundle config.
        iou_thres: NMS IoU threshold from the bundle config.
        timeout_s: Per-inference timeout the bundle config enforces.
        device: Device string from the bundle config.
        expected_item_count: How many items ``expected_items`` lists for this
            product and area, or ``None`` when the bundle does not declare any.
            Used to judge whether a benchmark frame exercised postprocessing at
            production scale, since NMS cost scales with surviving candidates.
    """

    product: str
    area: str
    weights: str
    imgsz: tuple[int, int] = (640, 640)
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    timeout_s: float = DEFAULT_INFERENCE_TIMEOUT_S
    device: str = "cpu"
    config_path: str = ""
    expected_item_count: int | None = None
    extra: dict[str, object] = field(default_factory=dict)
