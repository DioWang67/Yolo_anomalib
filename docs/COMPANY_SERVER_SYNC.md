# Company server synchronization

Operator-visible status is documented in
[Operator manual](OPERATOR_MANUAL.md#71-伺服器同步標籤). Station configuration,
monitoring and recovery ownership are documented in
[Engineering manual](ENGINEERING_MANUAL.md#13-公司伺服器同步).

The inspection station is local-first. Every result and operator review is
committed to `Result/inspection_records.sqlite3` before any network request is
attempted. A network outage therefore does not block inspection or lose the
server upload intent.

## Station configuration

Add these keys to the active global `config.yaml`:

```yaml
inspection_sync_enabled: true
inspection_sync_endpoint: "https://inspection-api.company/api/v1/inspections"
inspection_sync_api_token_env: "YOLO11_INSPECTION_SYNC_TOKEN"
inspection_sync_timeout_seconds: 10
inspection_sync_interval_seconds: 30
inspection_sync_batch_size: 20
inspection_sync_max_attempts: 12
inspection_sync_allow_insecure_http: false
```

Provision the token outside the YAML file:

```powershell
setx YOLO11_INSPECTION_SYNC_TOKEN "<token issued by IT>"
```

Restart the GUI after changing the environment. Do not put API tokens in
`config.yaml`, source control, Excel reports, screenshots, or support logs.

## HTTP contract

The station sends:

```http
POST /api/v1/inspections
Authorization: Bearer <token>
Content-Type: application/json
Idempotency-Key: <inspection_id>
X-Inspection-Revision: <positive integer>
```

The JSON body has:

```json
{
  "schema_version": 1,
  "idempotency_key": "<inspection_id>",
  "revision": 2,
  "inspection": {},
  "predictions": [],
  "artifacts": []
}
```

The company API must:

1. authenticate the station and authorize its machine or production line;
2. enforce a unique key on `inspection_id`;
3. atomically insert or update only when the incoming revision is newer;
4. return a 2xx response only after the database transaction commits;
5. return the same successful outcome when the same idempotency key and
   revision are replayed;
6. validate `schema_version` and reject unsupported versions explicitly;
7. store server receive time separately from the station inspection time;
8. treat artifact paths as station-local references unless a separate file
   upload contract is introduced.

The current connector uploads metadata, predictions, review state, and artifact
paths. It does not upload image bytes. Image transfer should use a separate,
checksummed object-storage endpoint so a large image cannot hold the metadata
transaction open.

## Retry and concurrency behavior

- The outbox row is written in the same SQLite transaction as the inspection.
- Only one background worker sends records from a station.
- A lease allows recovery if the program closes during an upload.
- Exponential backoff protects the company API during an outage.
- `inspection_id` is the idempotency key.
- Every data or review change increments `revision`.
- A stale HTTP success cannot mark a newer local revision as synchronized.
- After the configured maximum attempts, the row becomes `dead` and is shown
  as a synchronization warning in the inspection-history page.

Inspect status:

```powershell
python -m tools.inspection_sync_admin --result-root Result
```

After IT fixes the endpoint, certificate, authentication, or server error,
explicitly requeue dead-letter records:

```powershell
python -m tools.inspection_sync_admin --result-root Result --retry-dead
```

## Production rollout

1. Deploy the company API to a test environment with a valid TLS certificate.
2. Confirm the API implements the idempotency and revision rules above.
3. Configure one pilot station and a station-scoped token.
4. Run `tools.production_preflight` in strict mode.
5. Save test inspections while online and confirm the GUI reports them synced.
6. Disconnect the network, save more inspections, and confirm detection
   continues while the pending count rises.
7. Restore the network and confirm the pending count returns to zero without
   duplicate server rows.
8. Repeat with an operator review change and confirm the newer revision wins.
9. Record the API version, certificate owner, token rotation owner, retention
   policy, and rollback contact in the release record.

Never point every production station at a new server implementation at once.
Complete one supervised pilot and a rollback rehearsal first.
