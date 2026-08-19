# Inspection database

For role-based procedures, see
[Operator manual](../manuals/OPERATOR_MANUAL.md#7-檢測紀錄) and
[Engineering manual](../manuals/ENGINEERING_MANUAL.md#11-檢測資料庫與-excel).

Every saved inspection is indexed in `Result/inspection_records.sqlite3`.
Images remain immutable files; SQLite stores their paths and searchable
metadata, avoiding duplicate image blobs.

Existing result folders can be indexed once with:

```powershell
python -m tools.rebuild_inspection_database --result-root Result
```

The index contains:

- `inspections`: product, station, machine, work order, camera, model version,
  artifact paths, current human review, and Training Set state.
- `ai_predictions`: class, confidence, bounding box, and optional mask data.
- `inspection_artifacts`: original, preprocessed, annotated, crop, mask, and
  heatmap paths.
- `review_events`: append-only operator review history.
- `maintenance_events`: backup and retention actions for audit and recovery.
- `inspection_sync_outbox`: durable company-server upload state, revision,
  retry lease, and last error.

## Schema and migration safety

The application owns a versioned schema (`schema_info.version`). On startup it
runs an integrity check and applies registered migrations inside an exclusive
transaction. An existing database is backed up before every migration. A
database created by a newer application is rejected instead of being
downgraded.

Backups are consistent SQLite online snapshots, so committed WAL records are
included. They are written under `Result/database_backups` and verified before
being published.

Create a manual backup:

```powershell
python -m tools.maintain_inspection_data --result-root Result --backup-only
```

Verify a backup without changing the active database:

```powershell
python -m tools.restore_inspection_database --result-root Result --backup <backup-file>
```

To restore, first close every running inference GUI that uses this Result
directory. The restore command creates another safety backup of the current
database before replacement:

```powershell
python -m tools.restore_inspection_database --result-root Result --backup <backup-file> --confirm-application-closed
```

## Evidence retention

Automatic daily database backup is enabled by default. Image cleanup is
disabled by default and can be enabled in `config.yaml` after the site approves
its retention policy:

```yaml
global:
  inspection_backup_interval_hours: 24
  inspection_retention_cleanup_enabled: false
  inspection_retention_pass_image_days: 30
  inspection_retention_fail_preprocessed_days: 90
  inspection_retention_fail_all_image_days: 180
```

Preview cleanup candidates (no files are changed):

```powershell
python -m tools.maintain_inspection_data --result-root Result
```

Apply the approved plan:

```powershell
python -m tools.maintain_inspection_data --result-root Result --apply
```

Only known image artifacts below the selected Result root are eligible.
Inspection metadata, review events, JSON, Excel reports, files outside Result,
and artifacts still referenced by a retained inspection are preserved.

Example reports:

```sql
-- Products with the most AI review failures
SELECT product, failure_category, COUNT(*) AS cases
FROM inspections
WHERE review_outcome = 'fail'
GROUP BY product, failure_category
ORDER BY cases DESC;

-- Machines or cameras associated with lighting failures
SELECT machine_id, station, camera_id, COUNT(*) AS cases
FROM inspections
WHERE failure_category = 'lighting_issue'
GROUP BY machine_id, station, camera_id
ORDER BY cases DESC;

-- Review failure rate by model version
SELECT model_version,
       COUNT(*) AS reviewed,
       SUM(review_outcome = 'fail') AS failed_reviews,
       ROUND(100.0 * SUM(review_outcome = 'fail') / COUNT(*), 2) AS fail_rate_pct
FROM inspections
WHERE review_outcome <> ''
GROUP BY model_version
ORDER BY model_version;
```

Set `machine_id`, `station_id`, `work_order`, and `camera_id` in the active
configuration. If `machine_id` is omitted, the application records the Windows
host name; other unknown values remain empty instead of being invented.

## Production preflight

Before releasing a station, run the storage, integrity, backup/restore, disk,
and company-sync checks:

```powershell
python -m tools.production_preflight --result-root Result --config config.yaml --backup-restore-drill --strict
```

Warnings are release blockers in `--strict` mode. Use strict mode when company
synchronization is part of the station rollout. If synchronization is
explicitly out of scope, run without strict mode and record a named acceptance
for the sync-disabled warning. Company synchronization is documented in
`docs/data/COMPANY_SERVER_SYNC.md`.
