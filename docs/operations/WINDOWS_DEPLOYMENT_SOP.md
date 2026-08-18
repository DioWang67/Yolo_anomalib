# Windows Deployment SOP

This SOP is for deploying `yolo11_inference` on a Windows inspection PC. It
focuses on running an already trained model and validating the runtime package.
Training and dataset curation remain outside this repo.

The production GUI can initiate the file-based handoff to the sibling
`Yolo11_auto_train` project. Operator steps are documented in
`docs/manuals/OPERATOR_MANUAL.md`; training, position gates and recovery ownership are
documented in `docs/manuals/ENGINEERING_MANUAL.md`.

## Scope

- Target: Windows production or pilot machine.
- Runtime: packaged `dist\yolo11_inference\yolo11_inference.exe` or a checked
  out repo with the `yolo_anomalib` Python environment.
- Hardware: Hikrobot/MVS industrial camera when physical capture is required.

## Preconditions

Confirm these before deployment:

- The product/area model package exists under `models\<product>\<area>\<type>\`.
- `config.yaml` or the model-level `config.yaml` points to the intended weights.
- Camera, lens, fixture, exposure, gain and lighting are physically locked for
  production validation.
- The target machine has write access to the configured `output_dir`.
- The operator knows whether this deployment is `pilot`, `supervised
  production`, or `unattended production`.

Do not start unattended production if the current product/area still has open
readiness `WARN` items without written engineering acceptance.

## 1. Build Or Receive The Release Bundle

From a development machine:

```powershell
build_exe.bat
```

The bundle is written to:

```text
dist\yolo11_inference\
```

The packaged executable is built from `GUI.py`, not `main.py`. That matters
because packaged diagnostic flags such as `--check-hikrobot-runtime` are handled
by `GUI.py`.

If deploying from source instead of the packaged bundle, use the project Python
environment:

```powershell
conda activate yolo_anomalib
python main.py --product <PRODUCT> --area <AREA> --type yolo --image path\to\image.jpg
```

## 2. Copy Runtime Files

Copy the complete release folder to the target machine:

```text
yolo11_inference\
  yolo11_inference.exe
  _internal\
  models\
  config.yaml
  Runtime\
  README.md
  docs\
```

Keep model and config relative paths unchanged unless the config is updated at
the same time. Do not copy only the exe.

## 3. Validate Runtime Packaging

From the unpacked release folder:

```powershell
.\yolo11_inference.exe --check-hikrobot-runtime
```

Expected result:

- bundled DLL/CTI/CLProtocol files are found;
- `MvCameraControl.dll` loads successfully;
- command exits with code `0`.

If it fails, treat it as a packaging or DLL search path issue before debugging
camera hardware.

## 4. Validate Camera Grab

With the camera connected and not held by another process:

```powershell
.\yolo11_inference.exe --check-camera-grab
```

Expected result:

- camera enumeration succeeds;
- camera opens;
- trigger mode is set to off;
- one frame is acquired;
- command exits with code `0`.

If this fails with network or no-data symptoms, run:

```bat
tools\diagnostics\diagnose_camera.bat
```

Then compare logs with `docs/operations/CAMERA_RUNTIME_DIAGNOSTICS.md`.

## 5. Validate Product Config

For PCBA pilot, run the readiness gate:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\production_readiness_check.py `
  --config models\<PRODUCT>\<AREA>\yolo\config.yaml `
  --product <PRODUCT> `
  --area <AREA> `
  --output-json readiness_report_<PRODUCT>_<AREA>.json
```

Blocking `FAIL` must be fixed. `WARN` can be accepted only with a written
engineering note for supervised pilot use.

For the PCBA1 helper flow:

```powershell
.\pcba.bat readiness A
.\pcba.bat readiness B
```

## 6. Smoke Test Inference

Using a saved image:

```powershell
python main.py --product <PRODUCT> --area <AREA> --type yolo --image path\to\golden.jpg
```

Using the GUI:

```powershell
python GUI.py
```

Using packaged runtime:

```powershell
.\yolo11_inference.exe
```

Fusion mode is available through GUI/API when both YOLO and Anomalib model
folders exist:

```text
models\<PRODUCT>\<AREA>\yolo\
models\<PRODUCT>\<AREA>\anomalib\
```

The current `main.py --type` CLI accepts only `yolo` and `anomalib`.

## 7. Evidence Collection

The normal GUI path is `檢測紀錄 > 匯出 Excel` for reports and
`工程設定 > 模型補訓` for reviewed retraining handoff. The commands below are
engineering/PCBA pilot alternatives, not the primary operator workflow.

After pilot inference has generated `Result\`, collect review evidence:

```powershell
.\pcba.bat collect --product <PRODUCT> --area <AREA> --include-pass `
  --start-time <ISO-8601> --end-time <ISO-8601> `
  --strict-evidence `
  --output-csv ..\release_artifacts\yolo11_inference\review_manifest_<PRODUCT>_<AREA>.csv `
  --output-json ..\release_artifacts\yolo11_inference\review_manifest_<PRODUCT>_<AREA>.json
.\pcba.bat summary <AREA> --product <PRODUCT> `
  --readiness-json readiness_report_<PRODUCT>_<AREA>.json `
  --review-manifest-csv ..\release_artifacts\yolo11_inference\review_manifest_<PRODUCT>_<AREA>.csv `
  --output-json ..\release_artifacts\yolo11_inference\pilot_acceptance_summary_<PRODUCT>_<AREA>.json `
  --output-md ..\release_artifacts\yolo11_inference\pilot_acceptance_summary_<PRODUCT>_<AREA>.md
```

Or run the one-step helper. Because it does not pass explicit manifest output
paths, it writes the deterministic hash-scoped defaults described below:

```powershell
.\pcba.bat pilot <AREA> --product <PRODUCT> --include-pass `
  --start-time <ISO-8601> --end-time <ISO-8601>
```

The explicit collect/summary example writes:

- `readiness_report_<PRODUCT>_<AREA>.json`
- `..\release_artifacts\yolo11_inference\review_manifest_<PRODUCT>_<AREA>.csv`
- `..\release_artifacts\yolo11_inference\review_manifest_<PRODUCT>_<AREA>.json`
- `..\release_artifacts\yolo11_inference\pilot_acceptance_summary_<PRODUCT>_<AREA>.json`
- `..\release_artifacts\yolo11_inference\pilot_acceptance_summary_<PRODUCT>_<AREA>.md`

The one-step helper instead writes `readiness_report_<PRODUCT>_<AREA>_<scope-hash>.json`
and `pilot_acceptance_summary_<PRODUCT>_<AREA>_<scope-hash>.{json,md}` in the
current directory. Its paired `review_manifest_<PRODUCT>_<AREA>_<scope-hash>.{csv,json}`
is written under the station review root configured by `workspace.yaml` (currently
`..\station_data\yolo11_inference`). Pass that exact CSV to any later standalone
`summary` command; do not substitute the global `review_manifest.csv`.

Operators must fill `review_label` and `review_note` before the data is used
for retraining or go/no-go decisions.

## 8. Common Failures

| Symptom | First Check | Likely Cause |
| --- | --- | --- |
| Chinese text is garbled in PowerShell | `Get-Content -Encoding utf8` | Console/codepage reading issue |
| `torch` DLL import error | active Python path | wrong conda environment |
| `jsonargparse` missing | `pip show jsonargparse` | dependency mismatch |
| `--check-hikrobot-runtime` unknown | command target | running `python main.py`, not packaged exe/`GUI.py` |
| `MV_E_NODATA` | camera diagnostics logs | firewall, NIC, trigger mode, or another process holding camera |
| no result images | `output_dir` and write permission | path or permission mismatch |
| preflight reports sync WARN | rollout scope | synchronization is disabled; obtain written acceptance or finish company API setup |
| position retraining has no eligible samples | position-only reviewed holdout | collect `position_false_reject` OK or pure `POSITION_SHIFT` confirmed-NG cases |

## Exit Criteria

Deployment is ready for supervised pilot when:

- runtime packaging preflight passes;
- camera grab preflight passes or image-only mode is explicitly accepted;
- product readiness has no blocking `FAIL`;
- golden board smoke test passes repeatedly;
- known NG samples fail with expected reason codes;
- rollback bundle and config are available.

The build copies the complete `docs` folder into the release. Before handoff,
verify both role manuals are present on the target machine:

- `docs\manuals\OPERATOR_MANUAL.md`
- `docs\manuals\ENGINEERING_MANUAL.md`

