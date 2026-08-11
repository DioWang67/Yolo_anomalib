"""Command-line entry point for deployment-blocking model acceptance."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from core.services.acceptance_gate import (
    AcceptanceGatePolicy,
    run_candidate_acceptance,
)
from core.services.color_revision_contract import (
    capture_candidate_color_revision_contract,
    color_revision_overrides,
    verify_active_color_revision_contract,
)
from core.services.model_acceptance import ModelIdentity
from core.station_data import load_station_data_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a candidate model against a frozen acceptance snapshot."
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--global-config", required=True)
    parser.add_argument("--color-revisions-root")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--product", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--inference-type", default="yolo")
    parser.add_argument("--candidate-version", default="candidate")
    parser.add_argument("--candidate-weight", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--color-model")
    parser.add_argument("--min-confirmed", type=int, required=True)
    parser.add_argument("--max-false-positives", type=int, required=True)
    parser.add_argument("--max-false-negatives", type=int, required=True)
    parser.add_argument("--max-regressions", type=int, default=0)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Allow unconfirmed samples in the frozen snapshot.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Do not fail solely because candidate inference returned errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    weight_path = _required_file(args.candidate_weight, "candidate weight")
    config_path = _required_file(args.candidate_config, "candidate config")
    color_path = (
        _required_file(args.color_model, "color model")
        if args.color_model
        else None
    )
    color_revisions_root = (
        Path(args.color_revisions_root).expanduser().resolve()
        if args.color_revisions_root
        else load_station_data_paths(args.project_root).color_revisions
    )
    color_revision_contract = capture_candidate_color_revision_contract(
        revisions_root=color_revisions_root,
        candidate_config_path=config_path,
        global_config_path=args.global_config,
        color_model_present=color_path is not None,
        product=args.product,
        area=args.area,
        inference_type=args.inference_type,
    )
    revision_overrides = color_revision_overrides(color_revision_contract)
    identity = ModelIdentity(
        version=str(args.candidate_version),
        sha256=_sha256_file(weight_path),
        runtime_config_sha256=_sha256_file(config_path),
        color_model_sha256=(
            _sha256_file(color_path) if color_path is not None else ""
        ),
    )
    policy = AcceptanceGatePolicy(
        min_confirmed=args.min_confirmed,
        max_false_positives=args.max_false_positives,
        max_false_negatives=args.max_false_negatives,
        max_regressions=args.max_regressions,
        require_all_confirmed=not args.allow_pending,
        require_no_errors=not args.allow_errors,
    )

    def report_progress(index: int, total: int, sample_id: str) -> None:
        if index == 1 or index == total or index % 25 == 0:
            print(
                f"[acceptance] {index}/{total} sample={sample_id}",
                flush=True,
            )

    def verify_color_revisions() -> tuple[str, ...]:
        verify_active_color_revision_contract(
            color_revision_contract,
            revisions_root=color_revisions_root,
        )
        return ()

    result = run_candidate_acceptance(
        project_root=args.project_root,
        models_root=args.models_root,
        global_config_path=args.global_config,
        color_revisions_root=color_revisions_root,
        dataset_root=args.dataset_root,
        snapshot_manifest_path=args.snapshot_manifest,
        report_path=args.report,
        product=args.product,
        area=args.area,
        inference_type=args.inference_type,
        model_identity=identity,
        policy=policy,
        color_revision_overrides=revision_overrides,
        include_active_color_revisions=False,
        color_revision_contract=color_revision_contract,
        color_revision_contract_validator=verify_color_revisions,
        progress_callback=report_progress,
    )
    if result.passed:
        print(f"[acceptance] PASSED report={result.report_path}", flush=True)
        return 0
    print(
        "[acceptance] BLOCKED " + "; ".join(result.failures),
        file=sys.stderr,
        flush=True,
    )
    return 2


def _required_file(raw_path: str, label: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
