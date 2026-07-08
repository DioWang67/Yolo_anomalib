# Documentation Index

This page is the entry point for project documentation. Start here when you are
not sure which document owns a topic.

All Markdown files are UTF-8. On Windows PowerShell, use `Get-Content -Encoding
utf8 <file>` if Chinese text appears garbled.

## Quick Paths

| Need | Read First | Then Read |
| --- | --- | --- |
| Run the project locally | `README.md` | `docs/TECH_GUIDE.md` |
| Understand module boundaries | `docs/MODULE_ARCHITECTURE.md` | `core/detection_system.py` docstring |
| Deploy on a Windows inspection PC | `docs/WINDOWS_DEPLOYMENT_SOP.md` | `docs/CAMERA_RUNTIME_DIAGNOSTICS.md` |
| Prepare a PCBA pilot | `docs/PCBA_PILOT_RUNBOOK.md` | `docs/PRODUCTION_GO_LIVE_CHECKLIST.md` |
| Operator command reference | `docs/PCBA_OPERATOR_COMMANDS.md` | `docs/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md` |
| Release or roll back runtime/model/config | `docs/RELEASE_ROLLBACK_SOP.md` | `docs/MODEL_VERSION_GUIDE.md` |
| Diagnose camera runtime issues | `docs/CAMERA_RUNTIME_DIAGNOSTICS.md` | `tools/diagnostics/diagnose_camera.bat` |
| Benchmark exported runtime artifacts | `docs/FIRMWARE_RUNTIME_PLAN.md` | `tools/runtime_benchmark.py` |
| Review security constraints | `docs/SECURITY.md` | `tests/test_security.py` |

## Document Ownership

### User-Facing Start Here

- `README.md`: project overview, install, local run commands, quick model/config
  examples, build entry points.
- `docs/TECH_GUIDE.md`: deep technical training guide. This is broad and useful
  for onboarding, but it is not the shortest production SOP.
- `docs/DOCUMENTATION_INDEX.md`: this file.

### Architecture And Development

- `docs/MODULE_ARCHITECTURE.md`: current runtime architecture, module
  responsibilities, sync/async inference flow, and state safety notes.
- `docs/SECURITY.md`: path validation, YAML safety, dependency hygiene, and
  security test coverage.
- `CHANGELOG.md`: release notes and migration notes.
- `PROJECT_MEMORY.md`: project decisions, assumptions, risks, and current
  product direction.
- `PROJECT_TODO.md`: open implementation and production readiness work.

### PCBA Pilot And Production Readiness

- `docs/PCBA_INSPECTION_PLAN.md`: inspection scope and known limits.
- `docs/PCBA_PILOT_RUNBOOK.md`: controlled pilot procedure from config readiness
  to review manifest and acceptance summary.
- `docs/PCBA_OPERATOR_COMMANDS.md`: short commands for operators using
  `pcba.bat`.
- `docs/PCBA_PILOT_ACCEPTANCE_TEMPLATE.md`: record template for golden board,
  known NG, dry-run, and go/no-go signoff.
- `docs/PRODUCTION_GO_LIVE_CHECKLIST.md`: production readiness checklist and
  current PCBA1 gate result.
- `docs/PROGRESS_LOG.md`: dated implementation/progress record.

### Deployment, Runtime, And Operations

- `docs/WINDOWS_DEPLOYMENT_SOP.md`: step-by-step deployment SOP for a Windows
  inspection PC.
- `docs/RELEASE_ROLLBACK_SOP.md`: release bundle, model/config promotion, and
  rollback procedure.
- `docs/CAMERA_RUNTIME_DIAGNOSTICS.md`: packaged Hikrobot runtime checks and
  on-site camera diagnostics.
- `docs/FIRMWARE_RUNTIME_PLAN.md`: runtime artifact split and benchmark criteria
  for firmware or constrained edge targets.
- `docs/MODEL_VERSION_GUIDE.md`: model naming, Git LFS, versioning and rollback
  rules.

## Current Production Caveat

The repository has enough documentation for controlled pilot work, but PCBA
must not be treated as unattended production-ready until the relevant product
config has:

1. no blocking readiness `FAIL`;
2. documented acceptance or fixes for readiness `WARN`;
3. golden board repeatability evidence;
4. known NG validation evidence;
5. a filled pilot acceptance record;
6. a rollback model/config path.

