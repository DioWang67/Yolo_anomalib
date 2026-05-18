# PCBA Pilot Acceptance Record

Use this record for each product/area before enabling unattended production decisions. A readiness check with no blocking `FAIL` is necessary, but it is not enough by itself.

## Scope

- Date:
- Product:
- Area:
- Fixture / camera ID:
- Model config path:
- Model weights path:
- Operator / engineer:

## Readiness Gate

Command:

```powershell
D:\miniconda\envs\yolo_anomalib\python.exe tools\production_readiness_check.py --config <config.yaml> --product <PRODUCT> --area <AREA> --output-json readiness_report.json
```

Result:

- Blocking `FAIL` count:
- `WARN` items:
- Engineering note for each accepted `WARN`:

## Golden Board Repeatability

Run repeated captures with the approved OK board under locked lighting, lens, exposure, focus, and fixture settings.

- Golden board ID:
- Capture count:
- PASS count:
- FAIL count:
- Max inference time:
- Average inference time:
- Evidence folder:
- Result:
  - [ ] Accepted
  - [ ] Rejected

## Known NG Validation

Use physical or reviewed image samples for each defect class that is in current scope.

| Defect case | Sample count | Expected reason code | Actual result | Accepted |
| --- | ---: | --- | --- | --- |
| Missing component |  | `MISSING` |  |  |
| Wrong component |  | `WRONG_COMPONENT` |  |  |
| Position shift |  | `POSITION_SHIFT` |  |  |
| Unexpected component |  | `UNEXPECTED_COMPONENT` |  |  |

## Dry Run

Run without blocking production decisions. Review all FAIL cases before accepting the config.

- Dry-run duration:
- Total images:
- PASS count:
- FAIL count:
- False positive count:
- False negative count:
- Uncertain count:
- Review manifest path:
- Exported review dataset path:

## Go / No-Go

- [ ] No blocking readiness `FAIL`.
- [ ] All `WARN` items have accepted engineering notes.
- [ ] Golden board repeatability is accepted.
- [ ] Known NG cases fail with expected reason codes.
- [ ] Dry-run review has acceptable false positive / false negative rates.
- [ ] Rollback config and weights are documented.

Decision:

- [ ] Go for supervised pilot
- [ ] Go for unattended production
- [ ] No-go

Approver:
