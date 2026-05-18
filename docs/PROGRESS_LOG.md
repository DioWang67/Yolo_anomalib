# Progress Log

## 2026-05-18

### Goal

Move `yolo11_inference` from a YOLO inference/validation tool toward a controlled PCBA production-pilot workflow.

### Completed

- Created durable project tracking:
  - `PROJECT_MEMORY.md`
  - `PROJECT_TODO.md`
- Confirmed branch state:
  - current branch: `firmware-runtime-optimization`
  - local `main` is already an ancestor of the current branch
- Added PCBA inspection planning and rollout docs:
  - `docs/PCBA_INSPECTION_PLAN.md`
  - `docs/PRODUCTION_GO_LIVE_CHECKLIST.md`
  - `docs/PCBA_PILOT_RUNBOOK.md`
  - `docs/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md`
  - `docs/PCBA_OPERATOR_COMMANDS.md`
- Added production-pilot helper commands:
  - `pcba.bat`
  - `tools/pcba_pilot.py`
- Added readiness gate:
  - `tools/production_readiness_check.py`
- Added pilot evidence tools:
  - `tools/collect_review_cases.py`
  - `tools/export_review_dataset.py`
  - `tools/pilot_acceptance_report.py`
- Added decision and traceability support:
  - inspection decision reason codes
  - model info in result payloads
  - inference time in result payloads
  - config snapshot output
  - NG failure crop output with stable names
- Added focused tests for:
  - decision engine
  - result serialization
  - failure crops
  - readiness gate
  - review manifest collection
  - review dataset export
  - pilot acceptance summary
  - short operator command wrapper

### Verified

- `models/PCBA1/A/yolo/config.yaml` and `models/PCBA1/B/yolo/config.yaml` had no blocking readiness `FAIL` when checked on 2026-05-18.
- Remaining readiness warnings still require physical validation before unattended production use.
- `.\pcba.bat readiness A` was verified from PowerShell.
- Focused affected test set passed:

```text
143 passed
```

### Current Production Status

- Engineering workflow is ready for a supervised controlled pilot.
- Unattended production use is not approved yet.
- `review_manifest.csv` produced 0 cases during the initial check because no new PCBA inference evidence had been generated in `Result/`.

### Remaining Before Go-Live

- Run real PCBA1 A/B OK board images through inference.
- Run known NG images through inference.
- Generate non-empty review manifest evidence.
- Fill operator `review_label` values.
- Generate pilot acceptance summaries for A and B.
- Decide whether to fix or explicitly accept the current position tolerance and missing-slot warnings based on golden/NG/dry-run results.
