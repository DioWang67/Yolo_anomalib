# PCBA Operator Commands

Use `pcba.bat` from the repository root. It automatically uses `D:\miniconda\envs\yolo_anomalib\python.exe` when available.

## Common Flow

Run readiness for A:

```powershell
.\pcba.bat readiness A
```

Run readiness for B:

```powershell
.\pcba.bat readiness B
```

Collect review cases after inference:

```powershell
.\pcba.bat collect
```

Collect PASS and FAIL cases for golden board review:

```powershell
.\pcba.bat collect --include-pass
```

Build pilot summary for A:

```powershell
.\pcba.bat summary A
```

Build pilot summary for B:

```powershell
.\pcba.bat summary B
```

## One-Step Pilot Evidence Summary

After inference has generated `Result/`, run:

```powershell
.\pcba.bat pilot A --include-pass
```

or:

```powershell
.\pcba.bat pilot B --include-pass
```

This runs readiness, collects review cases, and builds the pilot summary with default file names.

## Outputs

- `readiness_report_A.json`
- `readiness_report_B.json`
- `review_manifest.csv`
- `review_manifest.json`
- `pilot_acceptance_summary_A.json`
- `pilot_acceptance_summary_A.md`
- `pilot_acceptance_summary_B.json`
- `pilot_acceptance_summary_B.md`

## Still Required

The helper does not replace physical validation. You still need to run real OK/NG board images first, then fill `review_label` in `review_manifest.csv`.
