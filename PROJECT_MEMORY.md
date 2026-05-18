# Project Memory

This file records durable project context for future coding sessions. Keep it concise and update it when decisions, paths, workflows, or known issues change.

## Project Classification

- Class B: Business Application / Internal Service for the current production-pilot push.
- Reason: This repository still contains YOLO11 inference and validation tooling, but the active goal is controlled PCBA line deployment with traceable decisions and operator review.
- Engineering depth: keep modules simple and runnable, but require config validation, result traceability, focused tests, and explicit go/no-go checks.

## Project Context

- Repository: `yolo11_inference`
- Primary domain: YOLO / computer vision inference pipeline
- Target application: production-line PCBA inspection, starting with missing / wrong component detection.
- Expected focus areas:
  - model loading
  - image/video preprocessing
  - inference
  - postprocessing
  - result visualization / export
  - configuration and path management

## Inspection Strategy

- YOLO should be used as the primary detector for known components.
- Missing component detection must not rely only on raw YOLO output.
- Expected components should be defined by product-level configuration.
- Each detected component should be validated against expected item definitions.
- Position validation should compare detected boxes against expected ROI / center tolerance.
- Runtime / model errors must be separated from product defects.
- Production direction should prioritize inspection determinism and traceability over model complexity.

## Target Inspection Scope

- Initial scope:
  - missing components
  - wrong components
  - obvious position shift
  - low-confidence detection review
- Extended scope:
  - polarity error
  - solder defect
  - surface anomaly
  - contamination / scratch / burn mark

## Required Production Features

- Board alignment is required before stable PCBA position validation.
- ROI crop should be supported for small components.
- Output artifacts should include:
  - raw image reference
  - annotated image
  - crop images for failed components
  - JSON result
  - CSV summary
  - model version
  - config snapshot
  - decision reason codes
  - inference time

## Technical Direction

- YOLO: component detection, missing/wrong component checks.
- Anomalib: unknown visual anomaly and subtle defect detection.
- Traditional CV: color, shape, polarity marker, brightness, contour checks.
- OCR/template matching: text, label, silkscreen, part number validation.

## Current Assumptions

- External inputs such as image paths, model paths, config files, and output directories should be validated.
- Experimental code may stay simple, but inference and training logic should remain separate.
- Avoid hardcoded credentials, machine-specific absolute paths, or hidden global state.
- Prefer `pathlib`, type hints, and small focused functions in Python.

## Important Paths

Add confirmed project paths here as they become stable.

- Model weights:
  - `models/PCBA1/A/yolo/weights/best.onnx`
  - `models/PCBA1/B/yolo/weights/best.onnx`
- Input samples:
- Output directory:
- Config files:
  - `models/PCBA1/A/yolo/config.yaml`
  - `models/PCBA1/B/yolo/config.yaml`
- Test data:

## Runtime / Environment Notes

Add environment details only when verified.

- Python version: 3.10.18 in `D:\miniconda\envs\yolo_anomalib`
- Main dependencies:
- GPU / CUDA requirements:
- Known working command:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe -m pytest tests\test_yolo_inference_model.py
```

## Decisions

Record meaningful decisions with date, reason, and trade-off.

| Date | Decision | Reason | Trade-off |
| --- | --- | --- | --- |
| 2026-05-18 | Created project memory file | Preserve project context across sessions | Starts with assumptions until repo details are filled in |
| 2026-05-18 | Treat PCBA1 A/B as controlled-pilot candidates, not unattended production-ready configs | Readiness gate has no blocking FAIL, but position tolerance and missing-slot warnings remain | Pilot can proceed only with documented warning acceptance and physical validation |
| 2026-05-18 | Add pilot acceptance summary as a filesystem tool | Keeps pilot signoff traceable on offline inspection machines | It summarizes evidence but does not replace physical golden/NG validation |
| 2026-05-18 | Add `pcba.bat` and `tools/pcba_pilot.py` as the operator entrypoint | Avoids requiring operators to memorize long Python commands | Keeps commands simple but still requires review labels and real inspection evidence |

## Known Issues / Risks

- YOLO alone is not sufficient for all PCBA defects.
- Small components may require ROI-based high-resolution inspection.
- Solder defects and micro defects require additional optical setup and anomaly detection.
- Stable production use requires controlled lighting, fixed camera geometry, and alignment.
- Dataset size and distribution may limit model generalization.
- Confidence / IoU thresholds can strongly affect inference behavior.
- Path handling should be explicit to avoid writing outputs into unexpected locations.
- Visualization and inference output formats should stay traceable to input files.
- PCBA1 A uses `mode: iou` and `tolerance: 1.06`; readiness check interprets this as effective minimum IoU `0.0106`, which is too loose unless explicitly justified by measured line data.
- PCBA1 B uses `tolerance: 10.27` with `tolerance_unit: percent`; this is wide for production position validation and must be verified against fixture variation.
- PCBA1 A/B still need golden board repeatability, known NG validation, and dry-run review before unattended production use.

## Validation Checklist

Before considering inference changes complete:

- [ ] Model path validation is clear.
- [ ] Input path validation covers missing files/directories.
- [ ] Empty model outputs are handled.
- [ ] Output files are written to predictable locations.
- [ ] Key parameters such as confidence and IoU thresholds are configurable.
- [ ] At least one representative inference run or focused test has been performed.

## Open Questions

- Which YOLO runtime is the current standard for this repo?
- What are the canonical model weights and sample inputs?
- What output format is expected: annotated images, labels, JSON, CSV, or mixed?
- Are there deployment constraints such as CPU-only, GPU-only, or edge device runtime?
