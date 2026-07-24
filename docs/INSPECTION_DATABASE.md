# Inspection database

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
