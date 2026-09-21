# Deployment Preflight Checker (`tools/system_check`)

Run this on a Windows machine **before** you trust it with production
inspection. It answers four questions and writes them down:

1. Can this machine run `yolo11_inference` at all?
2. Which hardware or software requirements are not met?
3. How fast does the real inference path actually run here?
4. How does that compare with the development or reference machine?

It is read-only. It installs nothing, downloads nothing, and does not touch the
registry, the firewall, system environment variables, production configuration,
or inspection evidence.

---

## Quick start

**Double-click `system_check.exe`.** That is the whole procedure. No arguments,
no typing. It finds the installation, runs every check, benchmarks the real
model, writes both reports next to itself, and keeps the window open until you
press Enter.

For that to work, put the `system_check` folder **beside** the application:

```text
D:\deployment\
├── system_check\          <- copy the whole folder; double-click the .exe inside
│   └── system_check.exe
└── yolo11_inference\      <- found automatically
    ├── yolo11_inference.exe
    └── models\
```

The checker searches its own folder, then up to three levels of parent
directories and their immediate subfolders, for something that looks like an
installation (`config.yaml`, `yolo11_inference.exe`, `Runtime\`, `models\`).
Use `--app-root` only if it guesses wrong or the app lives somewhere unrelated.

Everything below is optional, for people who want more than the default run.

```powershell
# From a source checkout, using the application's environment
D:\miniconda\envs\yolo_anomalib\python.exe -m tools.system_check

# Point at an installation explicitly
.\system_check.exe --app-root "D:\yolo11_inference"
```

Two files land in `--output-dir` (the current directory by default):

| File | Audience |
| --- | --- |
| `system_report.txt` | The person standing at the station |
| `system_report.json` | Tooling, and `--baseline` on the next machine |

### Recording and comparing a baseline

```powershell
# On the reference machine
system_check.exe --save-baseline baseline.json

# On the target machine, with the same baseline.json copied across
system_check.exe --baseline baseline.json
```

Both machines should use the same benchmark backend and the same model bundle.
When they do not, the comparison says so rather than printing numbers that look
comparable and are not.

---

## What it checks

| Check | Question it answers | Verdict basis |
| --- | --- | --- |
| `os.platform`, `os.architecture`, `os.build` | 64-bit Windows? | FAIL off Windows or off x64. No minimum build is stated anywhere in the repo, so the build is reported, not judged. |
| `cpu.model`, `cpu.cores`, `cpu.instruction_set` | Enough cores for the pipeline? | Advisory floor of 4 logical cores, derived from the 3 pipeline worker threads plus the GUI thread. Labelled SUGGESTED. |
| `memory.total`, `memory.available` | Will the pipeline's buffers fit? | Judged against this station's own `image_queue_max_mb`, not an invented minimum. |
| `disk.result`, `disk.application` | Room to write inspection evidence? | FAIL below `min_free_disk_mb` — a real threshold the pipeline enforces. |
| `gpu.present`, `gpu.cuda`, `gpu.ort_providers` | Is there a GPU, and can anything use it? | Informational. This project ships CPU-only wheels; a missing GPU is never a failure. Asks `torch.cuda.is_available()`, not just `nvidia-smi`. |
| `python.version`, `deps.*` | Right interpreter and packages? | FAIL on the five pins `start.bat` already refuses to launch on; WARNING on the rest. Three inspection modes — see `dependency_check.py`. The anomalib group is checked whenever those packages are **present**, not only when `enable_anomalib` is true: the build collects them regardless, so skipping on the flag hides drift in a shipped artifact. |
| `runtime.onnx_provider` | Will `.onnx` models load? | Reproduces the gate in `core/runtime_preflight.py`. |
| `runtime.vcredist`, `runtime.vc120` | Are the C++ runtimes resolvable? | Requirement taken from the binaries' **PE import tables**, not from filenames. ONNX Runtime needs the 2015-2022 CRT and does not ship it, so absence is FAIL. The Hikrobot SDK needs the 2013 CRT and *does* ship it inside `Runtime`, so the check looks there before concluding. |
| `runtime.hikrobot_files`, `runtime.hikrobot_load` | Is the camera SDK deployable? | Checks the same eight files as `GUI.py --check-hikrobot-runtime`, then actually loads the DLL. |
| `runtime.models` | Are the weights on this machine? | Models are external data and are **not** inside the executable. |
| `camera.device` | Is a camera attached? | Enumerates only. Opening a camera takes exclusive access and would disturb a running line. |
| `serial.light` | Is the LED controller reachable? | Optional; never opens the port. |
| `network.sync` | Can the outbox reach the server? | SKIP unless `inspection_sync_enabled`. Resolves and TCP-connects; sends no HTTP request and never reads the API token. |
| `benchmark.latency` | Does inference fit the timeout? | **FAIL** when P99 meets or exceeds the configured per-inference timeout — exceeding it fails a real inspection. |
| `benchmark.throughput` | How many inferences per second? | Reported, never judged: no FPS or cycle-time requirement exists in the repository. |
| `benchmark.memory` | Does the real footprint fit? | Measured peak working set against measured available memory. |

### Statuses

`PASS`, `WARNING`, `FAIL`, `UNKNOWN`, and `SKIP`.

`SKIP` is an addition to the four requested states, and it earns its place:
"server sync is switched off here" and "the GPU query failed" must not print
the same word, because they lead an operator to different actions.

Rolled up: any `FAIL` fails the machine; `WARNING` or `UNKNOWN` downgrades to
**PASS WITH WARNINGS**, because an undetermined requirement is not evidence of
compliance.

---

## The benchmark

Not a synthetic proxy. It loads the weights this station's model bundle names,
letterboxes at the configured `imgsz` using the same transform as
`core/utils.ImageUtils.letterbox`, runs the ONNX session, and applies the
bundle's own `conf_thres` and `iou_thres`.

- **Stages are timed separately** — preprocess, inference, postprocess, total —
  because they scale with different things. Preprocessing tracks camera
  resolution (3072x2048 down to 640x640 is real work); inference tracks the
  model and the CPU; postprocessing tracks how many candidates clear the
  confidence threshold.
- **Warm-up runs are discarded**, matching the warm-up the application performs
  in `YOLOInferenceModel.initialize`.
- **Nothing is written.** No inspection record, no result directory, no model
  bundle is touched.

### Two input modes

| Mode | Input | Use it for |
| --- | --- | --- |
| Default | Deterministic synthetic frame at the configured camera resolution | Comparing two machines. Identical pixels everywhere, nothing to ship between sites. |
| `--benchmark-image <path>` | A real board photo, read-only | A representative postprocess measurement. |

The synthetic frame is built for comparability, not realism, and it matters
which one you are looking at. **Postprocess cost scales with how many
candidates clear the confidence threshold**, because NMS runs over the
survivors. Measured on the Cable1/A model:

| Detections | Postprocess |
| --- | --- |
| 1 (synthetic frame) | 0.07 ms |
| 6 (real PASS board) | 0.42 ms |
| 19 | 0.29 ms |
| 57 | 0.76 ms |

So the synthetic frame's postprocess figure is a **lower bound**. The report
says so: `benchmark.postprocess_load` compares the detections the frame
produced against the bundle's `expected_items` count and warns when it falls
short. In absolute terms the gap is well under a millisecond against ~33 ms of
inference, so it never changes a timeout verdict — but a number presented
without that caveat invites someone to size a line on it.

A production image is identified by **content hash**, not filename
(`file:board.jpg@fbb39dacf3dc`). Two sites can easily hold different files
called `golden.bmp`, and a comparison that silently paired them would present
two unrelated measurements as a difference. Use a *current* board: an image
taken before the station's exposure was corrected will not detect the way the
line does today.

### Two performance concepts, kept apart

| Concept | Stated in the repo? | Verdict |
| --- | --- | --- |
| **Inference timeout** — `core/config.py` `timeout`, enforced by `core/fusion_inference.py`, which fails the inspection on breach | Yes | PASS / WARNING / FAIL |
| **Cycle time / throughput** — a property of the production line | **No.** No FPS, takt time or throughput figure exists anywhere in the repository | Measured and reported, **no verdict** |

Throughput carries no check at all. A PASS would imply a threshold was met; a
permanent UNKNOWN would nag on every run about a requirement nobody has
written down. It appears in the measurements and in the "requirements this
report does not assert" section, and nowhere else.
`auto_trigger.inspection_cooldown_ms` bounds how often a trigger may fire; it
is not a throughput target and is not treated as one.

### Backends

| Backend | Uses | When |
| --- | --- | --- |
| `onnx` | `onnxruntime` directly | Default for `.onnx` weights — what production deploys. Works in the standalone executable. |
| `ultralytics` | `ultralytics.YOLO`, as `core/yolo_inference_model.py` does | Truest to production, including wrapper overhead. Needs torch, so source checkout only. Required for `.pt` weights. |

`--benchmark-backend auto` (the default) picks by artifact type. A baseline and
a target recorded with different backends are flagged as not comparable.

### Percentiles

P99 from fewer than 100 runs is the maximum wearing a label, and the report
says so. Use `--benchmark-runs 100` or more when the P99 figure itself matters.

---

## Interpreting the comparison

```
                              Baseline    This machine      Difference
  RAM                            32 GB         15.6 GB      -51% lower
  Total (P95)                 48.48 ms        57.16 ms     +18% slower
  Throughput                 26.67 FPS        20.1 FPS      -25% lower
```

Direction is declared per metric, never inferred from the sign: higher latency
is slower, higher throughput is faster. The JSON carries an explicit
`higher_is_better` flag and a `better` / `worse` / `same` / `unknown` verdict
per row.

**A slower machine is not automatically unacceptable.** Whether this machine
passes is decided by `benchmark.latency` against the configured inference
timeout, not by the comparison table. The comparison exists to explain *why* a
machine behaves differently.

---

## Command line

```
--app-root PATH            Installation to inspect (auto-detected otherwise)
--output-dir PATH          Where the two report files go (default: cwd)
--baseline PATH            Compare against a saved baseline
--save-baseline PATH       Also write this run as a baseline
--skip-benchmark           Environment checks only
--skip-cpu-benchmark       Skip the synthetic CPU probe
--benchmark-backend        auto | onnx | ultralytics
--benchmark-runs N         Timed runs (default 50)
--benchmark-warmup N       Discarded runs (default 5)
--benchmark-image PATH     Benchmark a real image instead of the synthetic frame
--product / --area         Benchmark a specific model bundle
--strict                   Exit non-zero on warnings as well as failures
--list-requirements        Print the requirements derived from the repository
--verbose                  Show requirement provenance per check
--quiet                    Write the files, print nothing
--debug                    Print tracebacks on tool errors
```

Exit codes: `0` PASS or PASS WITH WARNINGS, `1` FAIL, `2` the checker itself
could not run.

---

## Building the standalone executable

```powershell
tools\system_check\build_system_check.bat
```

Output: `dist\system_check\system_check.exe`. Copy the whole `system_check`
folder to the target machine; no Python installation is needed there.

The shipped build is produced from a **fresh virtual environment** holding only
`onnxruntime`, `numpy`, `opencv-python`, `PyYAML`, `psutil` and `pyinstaller` —
none of the application stack — so nothing from a developer's machine can leak
into it:

```powershell
py -3.11 -m venv C:\temp\sc_venv
C:\temp\sc_venv\Scripts\python -m pip install onnxruntime==1.23.2 numpy==1.26.4 `
    opencv-python==4.9.0.80 PyYAML==6.0.2 psutil pyinstaller
C:\temp\sc_venv\Scripts\python -m PyInstaller --clean --noconfirm `
    --distpath dist --workpath build\system_check `
    tools\system_check\system_check.spec
```

Two build-environment traps the spec already handles, both of which produced an
executable that died before reaching any project code:

- **conda CRT dependencies.** A conda interpreter keeps `libffi` (for
  `_ctypes`) and `libexpat` (for `pyexpat`) in `<prefix>/Library/bin`, which is
  only on PATH inside an activated environment. The spec puts those directories
  on PATH for the build so PyInstaller can resolve them either way. A plain
  CPython venv needs none of this and the code no-ops.
- **`multiprocessing.freeze_support()`.** ONNX Runtime and the BLAS behind
  numpy spawn workers; in a frozen build each child re-executes the exe with
  `--multiprocessing-fork` and would otherwise hit argparse and print usage
  errors over the operator's report. `__main__.py` calls it first, as `GUI.py`
  does.

Torch, Ultralytics, PyQt5 and anomalib are deliberately excluded. Bundling
torch alone would add over two gigabytes to a tool whose entire purpose is to
be copied onto a machine before anything else is installed. The checker is
designed around that exclusion:

- The benchmark uses `onnxruntime`, which is what production deploys.
- A packaged application's package versions are read from its `_internal`
  metadata rather than by importing them.
- `torch.cuda` and the Hikrobot bindings are imported lazily; when absent the
  report says SKIP or UNKNOWN and names the application command that answers
  the same question (`yolo11_inference.exe --check-camera-grab`).

Run the checker from the application's own Python environment when you want the
Ultralytics backend or a CUDA answer from torch.

---

## Deployment checklist

What has to travel to the target machine, and what does not. "Per station"
means the item is station-specific and must not simply be copied from another
machine.

| Item | Deploy? | Where it goes | Notes |
| --- | --- | --- | --- |
| **Checker exe** (`dist/system_check/`) | Yes — first, on its own | Anywhere (e.g. a USB stick) | Copy the **whole folder**, not just the .exe. ~193 MB. Run it *before* deploying anything else; it reports what the machine is missing. Not needed after commissioning. |
| **Application** (`dist/yolo11_inference/`) | Yes | The station's application directory | Copy the whole folder. Carries its own Python, torch, PyQt5 and the `Runtime/` SDK inside `_internal/`. |
| **Models** (`models/`) | Yes — **separately** | Beside `yolo11_inference.exe` | **Not bundled into the executable.** A build copied without `models/` starts and then fails to load any model. `runtime.models` checks this. |
| **MVS runtime** (`Runtime/`) | No — already inside the app | `_internal/Runtime/` | Bundled by `yolo11_inference.spec`. The Hikrobot **camera driver** is a separate matter: install MVS on the station for the camera to enumerate. `runtime.hikrobot_files` and `camera.device` check both halves. |
| **VC++ 2015-2022 x64** | **Yes — install on the machine** | System | `vc_redist.x64.exe`. ONNX Runtime imports `MSVCP140`, `MSVCP140_1`, `VCRUNTIME140`, `VCRUNTIME140_1` and does **not** ship them. Without it every `.onnx` model fails to load. |
| **VC++ 2013 (VC120)** | No | — | Imported by 14 binaries in `Runtime`, but the SDK **ships its own copies** alongside them and Windows resolves from the loading module's directory. Do not install it on the strength of a filename. |
| **`config.yaml`** | Yes, but **per station** | Application directory | Never copy another station's file wholesale. `config.local.yaml` holds the machine-specific overrides (exposure, gain, `output_dir`) and is gitignored for that reason. Exposure and illumination calibration are station-local and must be re-taken. |
| **Benchmark baseline** (`baseline.json`) | Optional | Beside the checker | Only needed to answer "how does this machine compare". Record it on the reference machine with `--save-baseline`, and use the **same benchmark backend and input image** on both sides or the comparison is not like-for-like. |
| **Benchmark image** | Optional, but recommended | Beside the checker | A current known-good board photo. Without it the benchmark uses the synthetic frame, whose postprocess figure is a lower bound (see below). Its content hash is recorded, so two sites cannot accidentally compare different images. |

## What this tool will not tell you

Printed in every report, and available via `--list-requirements`:

**Not stated anywhere in the repository, so never judged here**

- Minimum total RAM
- Minimum CPU core count (an advisory floor is offered, clearly labelled)
- Minimum CPU instruction set
- Any FPS, cycle-time or throughput requirement
- Minimum VRAM (not applicable while the pinned wheels are CPU-only)
- A minimum Windows build

Where this tool offers a number for one of these, it is marked **SUGGESTED**
and the report says what it was derived from. None of them is a contract.

---

## Design notes

- **Decoupled from the application.** Nothing here imports `core`. The one
  piece of production logic it reproduces — letterbox preprocessing — is pinned
  by `tests/test_system_check_benchmark.py`, which imports both and asserts the
  arrays are identical. If production preprocessing changes, that test fails
  rather than the benchmark quietly measuring the wrong transform.
- **Exception isolation.** Every check runs through `run_check`, which turns any
  exception into an `UNKNOWN` result carrying the exception text.
  `KeyboardInterrupt` is deliberately not caught, so the tool stays abortable.
- **No identifying detail.** The report carries no host name, user name or
  serial number. A preflight report is normally e-mailed between sites, and it
  should not be the thing that carries identifying detail out of the plant. The
  sync API token is checked for presence only; its value is never read.
- **Every threshold is traceable.** `spec.py` records a `source` for each
  requirement, pointing at the file and lines it came from.

## Related tools

| Tool | Scope |
| --- | --- |
| `tools/check_runtime_environment.py` | The launcher's own fast gate: Python version, imports, five pinned versions. `start.bat` runs it on every launch. |
| `tools/production_preflight.py` | Result storage, backups and server sync readiness. |
| `tools/production_readiness_check.py` | Whether one config is ready for controlled production use. |
| `tools/runtime_benchmark.py` | Ad-hoc latency measurement over a directory of real images. |
| `verify_build.py` | Whether a PyInstaller build is complete. |
| **`tools/system_check`** | Whether a *machine* can run the application, and how fast. |
