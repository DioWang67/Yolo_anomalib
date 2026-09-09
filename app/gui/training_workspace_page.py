"""In-place page for the production retraining workflow.

The retraining surface used to be reached by launching the training tool as a
detached process through ``cmd.exe``, which put a console window and a second
application window in front of an operator who was already looking at the
inspection screen. This page moves that surface into the main window's
workspace stack, so retraining is a place you navigate to and come back from,
the way inspection records and engineering settings already are.

What it does *not* do is run the training. The work stays in a child process,
and that is load-bearing rather than incidental: the training entry point drops
its whole process to ``BELOW_NORMAL`` and caps the numerical thread pools so an
unattended retrain cannot starve the line, and its pipeline runs on an
in-process Qt thread. Hosting that inside the inspection process would put a
YOLO training run behind the same GIL and at the same scheduling priority as
the inspection loop. So the page is a view onto the shared job status.

The status view itself is :class:`ModelUpdateStatusDialog` in its embedded
mode, not a reimplementation. That class already owns job discovery, the
refresh cadence, resume and cancel (including starting the worker without a
console window), the workflow-step mapping and its own bilingual labels; a
second surface reading the same ``status.json`` would be a second place for
those rules to drift.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import normalize_language, tr
from app.gui.model_update_status_dialog import ModelUpdateStatusDialog


class TrainingWorkspacePage(QWidget):
    """Workspace page hosting the retraining status view."""

    back_to_inspection_requested = pyqtSignal()

    def __init__(
        self,
        *,
        data_root: str | Path,
        language: str = "zh",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._data_root = Path(data_root).expanduser().resolve()
        self._language = normalize_language(language)
        self._product = ""
        self._area = ""
        self._status_view: ModelUpdateStatusDialog | None = None
        self._build_ui()
        self._rebuild_status_view()

    # ------------------------------------------------------------------
    # Page contract
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.back_button = QPushButton()
        self.back_button.setObjectName("trainingBackButton")
        self.back_button.clicked.connect(self.back_to_inspection_requested.emit)
        header.addWidget(self.back_button)
        self.target_label = QLabel()
        self.target_label.setStyleSheet(
            "color: #245b8f; font-size: 11pt; font-weight: 600;"
        )
        header.addStretch()
        header.addWidget(self.target_label)
        layout.addLayout(header)

        self._status_container = QVBoxLayout()
        self._status_container.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._status_container, 1)

        self.launch_hint = QLabel()
        self.launch_hint.setWordWrap(True)
        self.launch_hint.setStyleSheet("color: #6b7280; font-size: 9pt;")
        layout.addWidget(self.launch_hint)

        self._apply_language()

    def set_language(self, language: str) -> None:
        """Switch this page, and the status view it hosts, to ``language``.

        The hosted view builds every label once at construction, so it is
        rebuilt rather than re-translated in place. That costs the current
        row selection, which is the honest trade for not maintaining a second
        translation path through forty widgets that would silently fall behind
        the first.
        """
        normalized = normalize_language(language)
        if normalized == self._language:
            return
        self._language = normalized
        self._apply_language()
        self._rebuild_status_view()

    def _apply_language(self) -> None:
        self.back_button.setText(tr(self._language, "training_back"))
        self.launch_hint.setText(tr(self._language, "training_page_hint"))
        self._refresh_target_label()

    def _refresh_target_label(self) -> None:
        if self._product or self._area:
            self.target_label.setText(
                f"{self._product or '—'} / {self._area or '—'}"
            )
        else:
            self.target_label.setText(tr(self._language, "training_no_target"))

    def show_for_target(self, product: str, area: str) -> None:
        """Point the page at the product and station the operator selected."""
        self._product = str(product or "").strip()
        self._area = str(area or "").strip()
        self._refresh_target_label()
        self._rebuild_status_view()

    # ------------------------------------------------------------------
    # Hosted status view
    # ------------------------------------------------------------------
    def _rebuild_status_view(self) -> None:
        if self._status_view is not None:
            self._status_view.shutdown()
            self._status_container.removeWidget(self._status_view)
            self._status_view.setParent(None)
            self._status_view.deleteLater()
            self._status_view = None

        view = ModelUpdateStatusDialog(
            data_root=self._data_root,
            language=self._language,
            selected_product=self._product or None,
            selected_area=self._area or None,
            background_refresh=True,
            embedded=True,
            parent=self,
        )
        self._status_view = view
        self._status_container.addWidget(view)

    def shutdown(self) -> None:
        """Release the hosted view.

        Any worker processes it started are deliberately left running: they
        are the retraining, they publish their own status, and closing the
        inspection GUI must not abandon a half-finished model update.
        """
        if self._status_view is not None:
            self._status_view.shutdown()

    def closeEvent(self, event) -> None:  # pragma: no cover - GUI lifecycle
        self.shutdown()
        super().closeEvent(event)
