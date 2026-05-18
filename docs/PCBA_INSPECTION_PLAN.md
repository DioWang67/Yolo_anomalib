# PCBA Inspection Plan

This project is currently a YOLO-based inference and validation tool. The near-term production direction is to make missing, wrong, and shifted component decisions deterministic and traceable before expanding into subtle visual defects.

## Classification

- Class C moving toward Class B.
- Reason: the codebase supports CV inference experiments today, but PCBA inspection needs repeatable product configuration, validation rules, and output traceability.
- Engineering depth: keep modules focused and testable without introducing enterprise-style abstractions.

## P0 Scope

The first stable inspection target is:

- missing components
- wrong components occupying expected slots
- obvious position shift
- unexpected components

The current decision layer reports these as machine-readable reason codes through `decision.reasons`.

## Alignment Direction

The current implementation uses translation-based expected-layout alignment derived from matched detections. This is a practical first step for fixture drift and small board offsets.

Use this only when:

- camera and fixture geometry are fixed
- board rotation/perspective error is small
- enough trusted components are detected to estimate a shared shift

Escalate to fiducial-based affine or homography alignment when:

- the board rotates between captures
- left/right sides show different offsets
- local position errors remain high after shared translation correction
- small components fail due to perspective or lens distortion

## Expected Item Configuration

Product configuration should define both count expectations and position expectations:

- `expected_items`: class list used for missing / unexpected class checks
- `position_config.expected_boxes`: measured expected ROIs per component
- `position_config.alignment`: shared shift estimation settings
- `position_config.missing_slot_check`: conservative ROI occupancy refinement

See `configs/products/pcba_example.yaml` for a minimal example.

## Traceability Requirements

Every failed inspection should preserve enough evidence to answer why it failed:

- final status
- `decision.reasons`
- missing item list
- unexpected item list
- slot mismatch details
- layout alignment metadata
- annotated image path
- crop paths when available
- failure crop paths for missing, wrong component, and position shift cases
- config snapshot path
- model weights path
- parsed model version when filename contains one
- confidence / IoU / image size thresholds
- product / area

## Known Limits

YOLO alone should not be treated as a complete AOI solution. Solder defects, polarity marks, scratches, contamination, and OCR-like checks need ROI-level second-stage logic such as Anomalib, traditional CV, OCR, or template matching.

## Review Data Collection

Use the review manifest tool after inspection runs to collect NG cases for human review:

```powershell
python tools\collect_review_cases.py --result-root Result --output-csv review_manifest.csv --output-json review_manifest.json
```

The manifest intentionally leaves `review_label` and `review_note` empty. Fill them during manual review with labels such as:

- `confirmed_ng`
- `false_positive`
- `false_negative`
- `uncertain`

This keeps production feedback usable for threshold tuning, dataset curation, and retraining decisions.

## Pilot Runbook

Use `docs/PCBA_PILOT_RUNBOOK.md` as the operating sequence before production pilot. The important gate is:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\production_readiness_check.py --config <config.yaml> --product <PRODUCT> --area <AREA>
```

The current repository root `config.yaml` is not PCBA-ready because it does not define product/area, expected items, position config, or expected boxes.
