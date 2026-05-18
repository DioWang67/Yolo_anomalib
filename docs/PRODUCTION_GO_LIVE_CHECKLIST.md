# Production Go-Live Checklist

This checklist targets controlled PCBA inspection rollout. Passing it means the system is ready for pilot production, not that all AOI defect classes are solved.

## Required Before Pilot

- [ ] Product and area names are fixed.
- [ ] Model weights exist on the production machine.
- [ ] `expected_items` matches the BOM / inspection scope.
- [ ] `position_config.<product>.<area>.enabled` is true.
- [ ] `expected_boxes` are measured from the real fixture and camera setup.
- [ ] `save_original`, `save_annotated`, and `save_crops` are enabled.
- [ ] `fail_on_unexpected` is enabled.
- [ ] Camera, lens, lighting, and fixture settings are locked.
- [ ] Golden board images pass consistently.
- [ ] Known NG samples fail with correct reason codes.
- [ ] At least one shift-long dry run has review manifest output.
- [ ] `docs/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md` is filled for each product/area.
- [ ] Operator review labels are defined: `confirmed_ng`, `false_positive`, `false_negative`, `uncertain`.
- [ ] Rollback model/config path is documented.

Run the config gate:

```powershell
python tools\production_readiness_check.py --config config.yaml --product PCBA --area TOP --output-json readiness_report.json
```

Blocking `FAIL` checks should be resolved before production use. `WARN` checks can be accepted only with an explicit engineering note.

The repository root `config.yaml` is a base LED-oriented config and does not currently pass the PCBA readiness gate. For PCBA pilot, create a real product config first; see `docs/PCBA_PILOT_RUNBOOK.md`.

## Current PCBA1 Gate Result

Checked on 2026-05-18:

- `models/PCBA1/A/yolo/config.yaml`: no blocking `FAIL`.
- `models/PCBA1/B/yolo/config.yaml`: no blocking `FAIL`.

Remaining warnings:

- PCBA1 A effective IoU tolerance is `0.0106`, which is too loose unless accepted by measured pilot data.
- PCBA1 A has `missing_slot_check` disabled.
- PCBA1 B position tolerance is `10.27%`, which is wide for production position validation.
- PCBA1 B has no `missing_slot_check` configured.

Do not mark unattended production ready until these warnings are either fixed or explicitly accepted after golden board, known NG, and dry-run validation.

## Feedback Loop

1. Collect review manifest:

```powershell
python tools\collect_review_cases.py --result-root Result --output-csv review_manifest.csv --output-json review_manifest.json
```

2. Fill `review_label` and `review_note`.

3. Export reviewed images for annotation:

```powershell
python tools\export_review_dataset.py --manifest-csv review_manifest.csv --output-dir ..\Yolo11_auto_train\data\pcba_review
```

4. Annotate `raw/images` and `raw/labels`, then run the Yolo11_auto_train pipeline.

5. Build the pilot acceptance summary:

```powershell
python tools\pilot_acceptance_report.py --product PCBA --area TOP --readiness-json readiness_report.json --review-manifest-csv review_manifest.csv --output-json pilot_acceptance_summary.json --output-md pilot_acceptance_summary.md
```
