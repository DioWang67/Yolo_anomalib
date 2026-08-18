# PCBA Pilot Acceptance Record

Use this record for each product/area before enabling unattended production decisions. A readiness check with no blocking `FAIL` is necessary, but it is not enough by itself.

## Scope

- Date:
- Product:
- Area:
- Fixture / camera ID:
- Model config path:
- Model weights path:
- Inference Git SHA (40 characters):
- Training Git SHA (40 characters):
- Git worktrees clean at pilot start/end:
- Runtime config SHA-256:
- Model weights SHA-256:
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
- Review manifest SHA-256:
- Exported review dataset path:
- Pilot started at:
- Pilot completed at:
- Shift duration:
- Live production cycle count (minimum 500 and at least one complete shift):

## Go / No-Go

- [ ] No blocking readiness `FAIL`.
- [ ] All `WARN` items have accepted engineering notes.
- [ ] Golden board repeatability is accepted.
- [ ] Known NG cases fail with expected reason codes.
- [ ] Dry-run review has acceptable false positive / false negative rates.
- [ ] Rollback config and weights are documented.
- [ ] Preflight evidence and backup/restore drill are attached by SHA-256;
      `--strict` was used when company sync is in rollout scope.
- [ ] Inspection-history XLSX was reopened and spot-checked against SQLite.
- [ ] Duplicate-suppression PASS candidates are all reviewed as `confirmed_ok`.
- [ ] Camera disconnect/reconnect was exercised on this exact code/artifact identity.
- [ ] Company sync is either explicitly out of scope or has verified receipt,
      reconnect, idempotency, and zero-pending-outbox evidence.

## Evidence Bundle and Authenticated Approval

- Evidence bundle/report path:
- Evidence bundle/report SHA-256:
- Protected PR or signed approval record URL:

Every approval must reference the same evidence SHA-256. A local CLI field or
typed name is not approval; record identity and decision in an authenticated PR
review or externally signed record.

| Required role | Authenticated identity | Approved at (ISO-8601) | Decision | Review/signature URL |
| --- | --- | --- | --- | --- |
| Process owner |  |  |  |  |
| AI/ML owner |  |  |  |  |
| Software owner |  |  |  |  |
| Production line owner |  |  |  |  |

Decision:

- [ ] Go for supervised pilot
- [ ] Go for unattended production
- [ ] No-go

Legacy single approver field (not sufficient for unattended-production approval):

這份本機紀錄只屬於佐證，不是經身分驗證的核准。製程、AI、軟體及產線負責人
必須在受保護的 PR review 或外部簽章紀錄中，核准完全相同的 evidence SHA。
