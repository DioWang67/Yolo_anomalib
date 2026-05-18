# Project TODO

This list tracks the path from YOLO inference tooling toward a production-ready PCBA inspection workflow. Keep implementation small and verifiable; promote items only when there is real code or validated process behind them.

## Current Branch Status

- Branch checked on 2026-05-18: `firmware-runtime-optimization`
- Local `main` merge status: `main` is already an ancestor of the current branch.
- Current branch has commits not yet contained in local `main`.
- Existing untracked model artifacts were present before this TODO update and should not be mixed into unrelated commits.

## P0 - Reliable Missing / Wrong Component MVP

- [x] Create project memory file.
- [x] Record PCBA inspection direction and known limitations.
- [x] Confirm current branch already contains local `main`.
- [x] Add project TODO tracker.
- [x] Centralize inspection decision status and reason codes.
- [x] Define a product-level expected item / expected box example config.
- [x] Ensure missing, unexpected, wrong slot, and position shift cases are reported separately.
- [x] Verify JSON / CSV / annotated image outputs include enough traceability.
- [x] Add focused tests for the decision layer.

## P1 - Stronger PCBA Inspection

- [x] Promote current translation alignment notes into operator-facing documentation.
- [ ] Add ROI crop strategy for small or high-risk components.
- [ ] Define per-component confidence / tolerance rules where needed.
- [ ] Add Anomalib ROI scoring for non-YOLO visual defects.
- [x] Save failed component crops with stable naming.

## P2 - Production Readiness

- [x] Snapshot runtime config with every inspection output.
- [x] Record model version / weights path / threshold values in results.
- [x] Track inference time and failure reasons per frame.
- [x] Add false positive / false negative review workflow.
- [x] Define dataset/versioning process for OK and NG samples.
- [x] Add production deployment checklist for camera, lighting, and fixture stability.
- [x] Add PCBA pilot runbook and readiness gate documentation.
- [x] Add pilot acceptance record template for golden board, known NG, and dry-run signoff.
- [x] Add pilot acceptance summary tool for readiness and review-manifest evidence.
- [x] Add short operator command wrapper for pilot readiness, review collection, and summary.

## Current Production Status

- Root `config.yaml` is not PCBA production-ready; it is missing product/area, expected items, position config, and expected boxes.
- `models/PCBA1/A/yolo/config.yaml` and `models/PCBA1/B/yolo/config.yaml` pass the blocking readiness gate on 2026-05-18.
- PCBA1 A warning: `position_iou_tolerance` is very loose; `mode: iou` with `tolerance: 1.06` is interpreted as effective minimum IoU `0.0106`.
- PCBA1 A warning: `missing_slot_check` is disabled.
- PCBA1 B warning: `position_tolerance_percent` is wide at `10.27%`; verify this against fixture and camera variation before pilot.
- PCBA1 B warning: `missing_slot_check` is not configured.
- Code/config are structurally ready for controlled pilot, but unattended production use still requires golden board repeatability, known NG validation, and a dry run review pass.

## Review Workflow

- Run `python tools\collect_review_cases.py --result-root Result --output-csv review_manifest.csv --output-json review_manifest.json`.
- Fill `review_label` with one of: `confirmed_ng`, `false_positive`, `false_negative`, `uncertain`.
- Use `review_note` for operator comments, fixture notes, or retraining hints.
- Run `python tools\pilot_acceptance_report.py --product <PRODUCT> --area <AREA> --readiness-json readiness_report.json --review-manifest-csv review_manifest.csv`.

## Open Questions

- What is the canonical PCBA product/area naming convention?
- Which components require ROI-level second-stage inspection?
- Which output format is the production source of truth: JSON, CSV, Excel, or database?
- What is the acceptable false fail / escape rate for the first pilot line?
