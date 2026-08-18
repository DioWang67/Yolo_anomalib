# Production Go-Live Checklist

角色操作先閱讀：

- [操作者手冊](../manuals/OPERATOR_MANUAL.md)
- [工程維運手冊](../manuals/ENGINEERING_MANUAL.md)

## Station-wide preflight

Run this before product-specific acceptance:

```powershell
python -m tools.production_preflight --config config.yaml --backup-restore-drill
```

`FAIL` always blocks release. `WARN` requires a named engineering acceptance
for a supervised pilot. When company synchronization is in rollout scope, run
with `--strict`; it must return only `PASS`, and the offline/reconnect and
duplicate-prevention pilot in `docs/data/COMPANY_SERVER_SYNC.md` is mandatory:

```powershell
python -m tools.production_preflight --config config.yaml --backup-restore-drill --strict
```

要產生可稽核證據，請另外傳入 `--output-json <Result 樹外的 path>`。報告會以 atomic replace
寫入並記錄精確 config SHA-256。設定檢查為 PASS 不代表公司伺服器 live handshake
或重送冪等性已通過。

When company synchronization is explicitly outside the current rollout,
`company_sync_configuration: WARN` documents that limitation. It is not
permission to claim company-server integration is complete.

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
- [ ] `docs/pilot/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md` is filled for each product/area.
- [ ] Operator review labels are defined: `confirmed_ng`, `false_positive`, `false_negative`, `uncertain`.
- [ ] Operator and engineering owners have reviewed the current role manuals.
- [ ] Inspection database integrity and backup/restore drill pass.
- [ ] Excel export was opened and spot-checked against SQLite records.
- [ ] Company sync is either accepted as out-of-scope or passes the strict
      offline/reconnect/idempotency pilot.
- [ ] Rollback model/config path is documented.

Run the config gate:

```powershell
python tools\production_readiness_check.py --config config.yaml --product PCBA --area TOP --output-json readiness_report_PCBA_TOP.json
```

Blocking `FAIL` checks should be resolved before production use. `WARN` checks can be accepted only with an explicit engineering note.

The repository root `config.yaml` is a base LED-oriented config and does not currently pass the PCBA readiness gate. For PCBA pilot, create a real product config first; see `docs/pilot/PCBA_PILOT_GUIDE.md`.

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
python tools\collect_review_cases.py --product PCBA --area TOP `
  --start-time <ISO-8601> --end-time <ISO-8601> --include-pass --strict-evidence `
  --output-csv ..\release_artifacts\yolo11_inference\review_manifest_PCBA_TOP.csv `
  --output-json ..\release_artifacts\yolo11_inference\review_manifest_PCBA_TOP.json
```

2. Fill `review_label` and `review_note`.

3. Export reviewed images for annotation:

```powershell
python tools\export_review_dataset.py `
  --manifest-csv ..\release_artifacts\yolo11_inference\review_manifest_PCBA_TOP.csv `
  --output-dir ..\Yolo11_auto_train\data\pcba_review
```

4. Annotate `raw/images` and `raw/labels`, then run the Yolo11_auto_train pipeline.

5. Build the pilot acceptance summary:

```powershell
python tools\pilot_acceptance_report.py --product PCBA --area TOP `
  --readiness-json readiness_report_PCBA_TOP.json `
  --review-manifest-csv ..\release_artifacts\yolo11_inference\review_manifest_PCBA_TOP.csv `
  --output-json ..\release_artifacts\yolo11_inference\pilot_acceptance_summary_PCBA_TOP.json `
  --output-md ..\release_artifacts\yolo11_inference\pilot_acceptance_summary_PCBA_TOP.md
```

The product, area, and time window must identify the same pilot scope in every
command. Never build a station-specific summary from the unfiltered global
manifest. These commands run from `yolo11_inference`; omitting `--result-root`
uses the canonical path from the paired workspace manifest.

此命令採 fail-closed：`NO_GO` 與 `HOLD` recommendation 會回傳非零 exit code。
即使 exit code 為零，也只代表 `READY_TO_START_SUPERVISED_PILOT`；報告固定記錄
`operational_acceptance_status: NOT_CAPTURED` 與 `merge_eligible: false`。
