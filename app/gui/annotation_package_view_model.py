"""Presentation-only annotation package state for ProcessingBatchDialog."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tools.annotation_packages import load_annotation_package
from tools.annotation_tool_launcher import AnnotationToolLauncher, ManualAnnotationToolLauncher
from tools.processing_reports import ProcessingReport


@dataclass(frozen=True)
class AnnotationActionViewModel:
    accepted: bool
    title: str
    message: str
    completion_path: str = ""
    items: tuple[tuple[str, str], ...] = ()
    item_results: tuple[str, ...] = ()


@dataclass
class AnnotationPackageViewModel:
    visible: bool = False
    package_id: str = ""
    package_path: str = ""
    package_root: str = ""
    status: str = ""
    requires_resume: bool = False
    completion_path: str = ""
    _launcher: AnnotationToolLauncher | None = None
    _resume: Callable[[str, str], object] | None = None

    @classmethod
    def from_report(
        cls,
        report: ProcessingReport,
        *,
        launcher: AnnotationToolLauncher | None = None,
        resume: Callable[[str, str], object] | None = None,
    ) -> AnnotationPackageViewModel:
        result = next(
            (item for item in report.sample_results if item.action == "ANNOTATION_PACKAGE_CREATED"),
            None,
        )
        if result is None:
            return cls()
        package_path = str(result.metadata.get("package_path") or "")
        try:
            package = load_annotation_package(package_path)
            items = tuple(
                (item.sample_id, item.requested_operation.value)
                for item in package.items
            )
        except Exception:
            items = ()
        return cls(
            visible=bool(package_path), package_id=str(result.metadata.get("package_id") or ""),
            package_path=package_path, package_root=str(Path(package_path).parent),
            status=str(result.metadata.get("package_status") or "WAITING_FOR_OPERATOR"),
            requires_resume=bool(result.metadata.get("requires_resume")),
            items=items,
            _launcher=launcher or ManualAnnotationToolLauncher(), _resume=resume,
        )

    def launch_tool(self) -> AnnotationActionViewModel:
        if not self.visible or self._launcher is None:
            return AnnotationActionViewModel(False, "Annotation tool unavailable", "No annotation package is available.")
        package = load_annotation_package(self.package_path)
        result = self._launcher.launch(package)
        return AnnotationActionViewModel(
            result.status.value in {"LAUNCHED", "MANUAL"},
            "Annotation Tool", result.message,
        )

    def resume(self, reason: str) -> AnnotationActionViewModel:
        if self._resume is None:
            return AnnotationActionViewModel(
                False, "Resume unavailable",
                "Resume service is not configured; run the offline resume command for this package.",
            )
        result = self._resume(self.package_path, reason)
        self.status = result.status.value
        self.completion_path = str(result.report_path)
        self.item_results = tuple(
            f"{revision.sample_id}: SUCCESS / {revision.actual_operation.value}"
            for revision in result.successful_revisions
        ) + tuple(
            f"{failure.sample_id}: FAILED / {', '.join(failure.error_codes)}"
            for failure in result.failures
        )
        accepted = self.status in {"COMPLETED", "PARTIAL_FAILURE"}
        return AnnotationActionViewModel(
            accepted, "Annotation Resume",
            f"Annotation resume status: {self.status}", self.completion_path,
        )
