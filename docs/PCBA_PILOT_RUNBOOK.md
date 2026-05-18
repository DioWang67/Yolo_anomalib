# PCBA Pilot Runbook

This runbook is the controlled path from model deployment to production pilot. It assumes the current inspection scope is missing, wrong, unexpected, and shifted components.

For short operator commands, use `docs/PCBA_OPERATOR_COMMANDS.md`.

## Current Readiness Result

The repository root `config.yaml` is not production-ready for PCBA inspection. It currently points to an LED model and lacks PCBA production fields:

- no `current_product`
- no `current_area`
- no PCBA `expected_items`
- no enabled `position_config`
- no measured `expected_boxes`

Use `configs/products/pcba_example.yaml` as the structure reference, then create a real product config with measured values from the production fixture.

The current PCBA1 model configs were checked on 2026-05-18:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\production_readiness_check.py --config models\PCBA1\A\yolo\config.yaml --product PCBA1 --area A
D:\miniconda\envs\yolo_anomalib\python.exe tools\production_readiness_check.py --config models\PCBA1\B\yolo\config.yaml --product PCBA1 --area B
```

Both configs have no blocking `FAIL` result. They are controlled-pilot candidates, not yet approved for unattended production use.

Open warnings before pilot:

- PCBA1 A: `position_iou_tolerance` is very loose. `mode: iou` with `tolerance: 1.06` is interpreted as effective minimum IoU `0.0106`.
- PCBA1 A: `missing_slot_check` is disabled.
- PCBA1 B: `position_tolerance_percent` is `10.27%`, which must be justified against actual fixture and camera variation.
- PCBA1 B: `missing_slot_check` is not configured.

These warnings can be accepted only for a supervised pilot with a written engineering note and golden/NG validation evidence.

## Pilot Entry Criteria

Before starting pilot, the actual product config must pass:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\production_readiness_check.py --config <config.yaml> --product <PRODUCT> --area <AREA> --output-json readiness_report.json
```

Blocking `FAIL` results must be fixed before pilot. `WARN` results require a written engineering note in the pilot log.

## Required Product Config

At minimum, the production config must define:

```yaml
current_product: PCBA1
current_area: A

weights: models/PCBA1/A/yolo/weights/best.onnx
conf_thres: 0.4
iou_thres: 0.45
output_dir: ./Result
fail_on_unexpected: true
save_original: true
save_annotated: true
save_crops: true

expected_items:
  PCBA1:
    A:
      - R101
      - C205

position_config:
  PCBA1:
    A:
      enabled: true
      mode: center
      tolerance: 12
      tolerance_unit: pixel
      alignment:
        enabled: true
        min_source_count: 2
      missing_slot_check:
        enabled: true
      expected_boxes:
        R101:
          x1: 100
          y1: 120
          x2: 140
          y2: 150
        C205:
          x1: 220
          y1: 260
          x2: 250
          y2: 290
```

## Pilot Procedure

1. Deploy trained artifacts from `Yolo11_auto_train` to `yolo11_inference/models/<product>/<area>/yolo/`.
2. Create or update the product config with measured `expected_items` and `expected_boxes`.
3. Run `production_readiness_check.py`.
4. Run golden board validation.
5. Run known NG board validation.
6. Run a dry run using real line images without blocking production decisions.
7. Collect review manifest:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\collect_review_cases.py --result-root Result --output-csv review_manifest.csv --output-json review_manifest.json
```

8. Fill `review_label` and `review_note`.
9. Export reviewed cases to the training repo:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\export_review_dataset.py --manifest-csv review_manifest.csv --output-dir D:\Git\robotlearning\Yolo11_auto_train\data\pcba_review
```

10. Annotate exported images before retraining.
11. Generate the pilot acceptance summary:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\pilot_acceptance_report.py --product <PRODUCT> --area <AREA> --readiness-json readiness_report.json --review-manifest-csv review_manifest.csv --output-json pilot_acceptance_summary.json --output-md pilot_acceptance_summary.md
```

12. Fill `docs/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md` for each product/area before enabling supervised pilot or unattended production decisions.

## Go / No-Go Rules

Pilot can start when:

- readiness check has no `FAIL`
- golden board pass rate is stable across repeated captures
- known NG examples fail with the expected reason codes
- every FAIL output has annotated image, config snapshot, and NG crop evidence
- rollback model/config is available

Do not start unattended production use when:

- expected boxes are missing or copied from a different fixture
- board position varies beyond translation-only alignment
- lighting or camera settings are still being adjusted
- `review_manifest.csv` cannot be generated
- operators have no false positive / false negative review process

## Link With Yolo11_auto_train

`Yolo11_auto_train` already supports training, position validation, detection config export, and deployment into `yolo11_inference`. The feedback loop is:

```text
yolo11_inference Result/
  -> collect_review_cases.py
  -> review_manifest.csv
  -> operator labels review_label
  -> export_review_dataset.py
  -> Yolo11_auto_train/data/pcba_review/raw/images + raw/labels
  -> annotation
  -> training / validation / deploy
```

The export tool deliberately creates empty label placeholders. Operators or labelers must annotate them before retraining; the system must not auto-generate labels from NG crops.
