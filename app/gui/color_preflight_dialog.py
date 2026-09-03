"""Read a reference-board inspection and show what colour has left to spare.

The second step of the start-of-shift ritual, beside the illumination
calibration: brightness is a proxy, this is the measurement. It reads an
inspection the station has already saved rather than driving the camera, so it
sits outside the capture path entirely -- and shows how old that inspection is,
because the one real trap is judging today's shift on last week's board.

Nothing here decides anything. The evaluation lives in
``core.services.color_preflight_runner``, shared with ``tools/color_preflight.py``
so the dialog and the command cannot drift into different verdicts. Recording a
reference is the only write, it is named, and it never touches a baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gui.color_gamut_view import ColorGamutRow
from app.gui.i18n import normalize_language, tr
from core.services.color_preflight import (
    REMEDY_BASELINE,
    REMEDY_MARGIN,
    REMEDY_MISREAD,
    REMEDY_NO_REFERENCE,
    REMEDY_STALE_REFERENCE,
    STATE_BELOW_THRESHOLD,
    STATE_MARGIN_LOW,
    STATE_MISSING,
    STATE_NO_REFERENCE,
    STATE_OK,
    STATE_REFERENCE_STALE,
    STATE_UNEXPECTED,
    STATE_UNMEASURED,
    STATUS_NG,
    STATUS_OK,
    STATUS_WARN,
    ColorPreflightReport,
    remedy_causes,
)
from core.services.color_preflight_runner import (
    ColorPreflightSource,
    ColorPreflightUnavailable,
    gamut_samples_for,
    latest_inspection_snapshot,
    run_color_preflight,
)
from core.services.color_preflight_store import (
    ColorPreflightLedger,
    ColorPreflightRecord,
    ColorPreflightStoreError,
    record_reference_margins,
)

#: The application's own palette (see ``DetectionSystemGUI.init_ui``): a cool
#: neutral ground with a desaturated green and red. Reusing it rather than
#: bringing in a second set of colours is what keeps this panel looking like
#: part of the machine's software instead of a page from somewhere else. Only
#: the amber is added, mixed to sit with the existing green and red rather than
#: shout over them.
_INK = "#1f2933"
_INK_SOFT = "#52616f"
_INK_FAINT = "#829ab1"
_RULE = "#cbd5df"
_RULE_SOFT = "#dde5ec"
_SURFACE = "#ffffff"
_GROUND = "#f4f6f8"
_OK = "#16794c"
_OK_WASH = "#e6f1eb"
_WARN = "#8a5a10"
_WARN_WASH = "#fbf1de"
_NG = "#b42318"
_NG_WASH = "#fbeae8"

_STATUS_COLORS = {
    STATUS_OK: (_OK, _OK_WASH),
    STATUS_WARN: (_WARN, _WARN_WASH),
    STATUS_NG: (_NG, _NG_WASH),
}
_STATUS_VERDICT_KEYS = {
    STATUS_OK: "preflight_verdict_ok",
    STATUS_WARN: "preflight_verdict_warn",
    STATUS_NG: "preflight_verdict_ng",
}

#: How each state reads to an operator, and which severity it carries. The raw
#: identifiers are for the ledger; a panel at a machine should not ask anyone to
#: learn them.
_STATE_LABELS = {
    STATE_OK: ("preflight_state_ok", STATUS_OK),
    STATE_MARGIN_LOW: ("preflight_state_margin_low", STATUS_WARN),
    STATE_NO_REFERENCE: ("preflight_state_no_reference", STATUS_WARN),
    STATE_REFERENCE_STALE: ("preflight_state_reference_stale", STATUS_WARN),
    STATE_BELOW_THRESHOLD: ("preflight_state_below_threshold", STATUS_NG),
    STATE_MISSING: ("preflight_state_missing", STATUS_NG),
    STATE_UNEXPECTED: ("preflight_state_unexpected", STATUS_NG),
    STATE_UNMEASURED: ("preflight_state_unmeasured", STATUS_NG),
}

#: The product's own colours, for the swatch beside each row. A named row is
#: read; a coloured one is spotted, and this table is meant to be glanced at.
_SWATCHES = {
    "red": "#c0392b",
    "green": "#1e8449",
    "orange": "#d98324",
    "yellow": "#e8c33c",
    "black": "#2b3137",
    "white": "#f2f4f6",
    "blue": "#1f6feb",
    "brown": "#7a4a2b",
    "purple": "#6b3fa0",
    "gray": "#8899a6",
    "grey": "#8899a6",
}

#: How often to look for a newer inspection. Slow enough to be free, fast
#: enough that the table has updated by the time the operator looks back.
_WATCH_INTERVAL_MS = 1500

#: The shared remedy causes, rendered in this dialog's language. Which causes
#: apply is decided once in ``core.services.color_preflight``; this only names
#: them, so the command-line front end cannot give different advice.
_REMEDY_KEYS = {
    REMEDY_MISREAD: "preflight_remedy_misread",
    REMEDY_MARGIN: "preflight_remedy_margin",
    REMEDY_STALE_REFERENCE: "preflight_remedy_stale",
    REMEDY_NO_REFERENCE: "preflight_remedy_record",
    REMEDY_BASELINE: "preflight_remedy_baseline",
}

def _first_sentence(text: str) -> str:
    """The lead sentence, for a panel read in passing.

    The rest is kept as a tooltip rather than deleted: the reasoning is worth
    having the first time somebody meets a state, and in the way every shift
    after that.
    """
    lead = text.split("\n")[0].split("。")[0].strip()
    if not lead:
        return ""
    return lead if lead.endswith(("。", ".", "：")) else lead + "。"


def _swatch(color_name: str, size: int = 12) -> QPixmap:
    """A small chip of the product colour, or a hollow one when unknown."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing)
        fill = _SWATCHES.get(color_name.strip().casefold())
        painter.setPen(QColor(_RULE))
        painter.setBrush(QColor(fill) if fill else Qt.NoBrush)
        painter.drawEllipse(1, 1, size - 3, size - 3)
    finally:
        painter.end()
    return pixmap


class ColorPreflightDialog(QDialog):
    """Show the pre-shift colour standing for one station scope.

    Args:
        config_path: Live station ``config.yaml`` -- the reference is read from
            and recorded into it.
        results_root: Where saved inspections live.
        product / area / model_type: The scope being checked.
        ledger: Append-only log of runs.
        run_fn: Evaluation seam, ``run_color_preflight`` by default so tests can
            drive the dialog without a station on disk.
        record_fn: Reference-writing seam, ``record_reference_margins``.
        operator_prompt_fn: Returns ``(name, accepted)``; injected so the naming
            step can be exercised without a modal.
        language: UI language code.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        results_root: Path,
        product: str,
        area: str,
        model_type: str,
        ledger: ColorPreflightLedger,
        run_fn: Callable[..., tuple[ColorPreflightReport, ColorPreflightSource]] = (
            run_color_preflight
        ),
        record_fn: Callable[..., object] = record_reference_margins,
        operator_prompt_fn: Callable[[], tuple[str, bool]] | None = None,
        language: str = "en",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._results_root = Path(results_root)
        self._product = product
        self._area = area
        self._model_type = model_type
        self._ledger = ledger
        self._run_fn = run_fn
        self._record_fn = record_fn
        self._operator_prompt_fn = operator_prompt_fn or self._ask_operator
        self._language = normalize_language(language)
        self._report: ColorPreflightReport | None = None
        self._baseline_sha256 = ""

        self._seen_snapshot: Path | None = None
        self.setWindowTitle(self._t("preflight_title"))
        self.setMinimumWidth(620)
        self._build_ui()
        self.refresh()

        # Watch for the next inspection instead of asking the operator to go
        # away, run one, come back and press a button. The dialog is shown
        # non-modally for the same reason: the golden sample is already in the
        # fixture, so the natural move is to press 開始檢測 and watch this
        # update. Polling a path rather than hooking the pipeline keeps the
        # check out of the capture path, which is the whole point of reading a
        # saved result.
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(_WATCH_INTERVAL_MS)
        self._watch_timer.timeout.connect(self._poll_for_new_inspection)
        self._watch_timer.start()

    # ------------------------------------------------------------------
    def _t(self, key: str, **kwargs: object) -> str:
        text = tr(self._language, key)
        return text.format(**kwargs) if kwargs else text

    def _build_ui(self) -> None:
        # The verdict leads. It was at the bottom under the table, which is the
        # wrong way round for a panel read at a machine in passing: the one
        # thing an operator needs from across the bench is whether they may
        # run, and the detail is for when the answer is no.
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {_GROUND}; }}
            QLabel#preflightVerdict {{
                font-size: 15pt;
                font-weight: 600;
                padding: 12px 14px;
                border-radius: 6px;
            }}
            QLabel#preflightScope {{ color: {_INK_SOFT}; font-size: 9pt; }}
            QLabel#preflightMeta {{ color: {_INK_FAINT}; font-size: 8pt; }}
            QLabel#preflightNotice {{
                background-color: {_SURFACE};
                border: 1px solid {_RULE};
                border-radius: 6px;
                padding: 8px 10px;
                color: {_INK};
                font-size: 9pt;
            }}
            QLabel#preflightRemedy {{ color: {_INK_SOFT}; font-size: 9pt; }}
            QPushButton {{
                background-color: #eef2f6;
                color: {_INK};
                border: 1px solid {_RULE};
                padding: 8px 14px;
                border-radius: 6px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background-color: #e4ebf2; }}
            QPushButton#primaryAction {{
                background-color: {_OK};
                color: white;
                border: none;
            }}
            QPushButton#primaryAction:hover {{ background-color: #12643f; }}
            QPushButton#secondaryAction {{
                background-color: {_SURFACE};
                color: #243b53;
                border: 1px solid #bcccdc;
            }}
            QPushButton:disabled {{
                background-color: #d9e2ec;
                color: {_INK_FAINT};
                border-color: #d9e2ec;
            }}
            /* An id selector outranks a bare :disabled, so the primary action
               needs its own disabled rule or it stays green while unusable --
               and "you may not record this board" is the point. */
            QPushButton#primaryAction:disabled {{
                background-color: #d9e2ec;
                color: {_INK_FAINT};
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self._verdict_label = QLabel("")
        self._verdict_label.setObjectName("preflightVerdict")
        self._verdict_label.setWordWrap(True)
        layout.addWidget(self._verdict_label)

        self._scope_label = QLabel(
            self._t(
                "preflight_scope",
                product=self._product,
                area=self._area,
                model_type=self._model_type,
            )
        )
        self._scope_label.setObjectName("preflightScope")
        layout.addWidget(self._scope_label)

        self._notice_label = QLabel("")
        self._notice_label.setObjectName("preflightNotice")
        self._notice_label.setWordWrap(True)
        self._notice_label.setVisible(False)
        layout.addWidget(self._notice_label)

        # A column of pictures, not a table of numbers: the crop that was
        # measured, and where its pixels sit inside the envelope. See
        # ``app/gui/color_gamut_view.py`` for what each row carries.
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(6)
        self._rows_scroll = QScrollArea()
        self._rows_scroll.setWidget(self._rows_host)
        self._rows_scroll.setWidgetResizable(True)
        self._rows_scroll.setFrameShape(QScrollArea.NoFrame)
        self._rows_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(self._rows_scroll, 1)

        # Filled in per reading by ``_remedy_text``; empty until there is one.
        self._remedy_label = QLabel("")
        self._remedy_label.setObjectName("preflightRemedy")
        self._remedy_label.setWordWrap(True)
        self._remedy_label.setVisible(False)
        layout.addWidget(self._remedy_label)

        self._meta_label = QLabel("")
        self._meta_label.setObjectName("preflightMeta")
        self._meta_label.setWordWrap(True)
        layout.addWidget(self._meta_label)

        # Onboarding, not chrome: when there is a reading on screen the remedy
        # line already says what to do, and a standing instruction under it is
        # one more block of text to skip past.
        self._hint_label = QLabel(self._t("preflight_hint"))
        self._hint_label.setObjectName("preflightMeta")
        self._hint_label.setWordWrap(True)
        layout.addWidget(self._hint_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self._refresh_btn = QPushButton(self._t("preflight_refresh_btn"))
        self._refresh_btn.setObjectName("secondaryAction")
        self._refresh_btn.clicked.connect(self.refresh)
        self._record_btn = QPushButton(self._t("preflight_record_btn"))
        # The application's own primary-action styling, so the one button that
        # writes anything looks like every other committing button in the app.
        self._record_btn.setObjectName("primaryAction")
        self._record_btn.clicked.connect(self._on_record)
        self._record_btn.setEnabled(False)
        close_btn = QPushButton(self._t("preflight_close_btn"))
        close_btn.setObjectName("secondaryAction")
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(self._refresh_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        buttons.addWidget(self._record_btn)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-read the newest saved inspection and judge it."""
        try:
            report, source = self._run_fn(
                config_path=self._config_path,
                results_root=self._results_root,
                product=self._product,
                area=self._area,
                model_type=self._model_type,
            )
        except ColorPreflightUnavailable as exc:
            self._report = None
            self._record_btn.setEnabled(False)
            while self._rows_layout.count():
                item = self._rows_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self._meta_label.clear()
            self._meta_label.setToolTip("")
            self._hint_label.setVisible(True)
            self._notice_label.setVisible(False)
            self._show_verdict("", self._t("preflight_unavailable", error=exc))
            return

        self._report = report
        self._seen_snapshot = source.snapshot_path
        self._baseline_sha256 = source.baseline_sha256
        samples = gamut_samples_for(self._config_path, source.snapshot_path)
        self._render(report, source, samples)
        self._append_to_ledger(report, source)

    def _render(
        self,
        report: ColorPreflightReport,
        source: ColorPreflightSource,
        samples: dict,
    ) -> None:
        # Just the file name, with the full path as a tooltip: the path is 130
        # characters of mostly repeated prefix and it was crowding out the
        # reading. Counting path segments to pull the date out of it would be
        # both fragile and redundant -- the timestamp is on the next row.
        snapshot = source.snapshot_path
        age = source.age
        taken = source.taken_at[:19] + (f"（{age}）" if age else "")
        # One subdued line rather than a form competing with the data: which
        # inspection, when, and the bar it is being held to. The age stays
        # prominent enough to catch, because judging this shift on last week's
        # board is the one trap in reading a saved result.
        self._meta_label.setText(
            f"{self._t('preflight_taken_at')}: {taken}"
            f"　·　{self._t('preflight_retention_floor')}:"
            f" {report.minimum_margin_retention:.0%}"
        )
        self._meta_label.setToolTip(str(snapshot))
        self._hint_label.setVisible(False)

        notices = []
        if report.reference_is_stale:
            notices.append(self._t("preflight_reference_stale"))
        elif report.reference_is_missing:
            notices.append(self._t("preflight_no_reference"))
        if report.baseline_provenance_failure:
            notices.append(
                self._t("preflight_provenance")
                + "：" + report.baseline_provenance_failure
            )
        # One line on screen, the full wording on hover. These notices are
        # worth having and not worth three lines of the panel every shift.
        self._notice_label.setText(
            "　·　".join(_first_sentence(item) for item in notices)
        )
        self._notice_label.setToolTip("\n".join(notices))
        self._notice_label.setVisible(bool(notices))

        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for reading in report.colors:
            state_key, severity = _STATE_LABELS.get(
                reading.state, (reading.state, STATUS_WARN)
            )
            stroke, wash = _STATUS_COLORS[severity]
            self._rows_layout.addWidget(
                ColorGamutRow(
                    reading=reading,
                    sample=samples.get(reading.color),
                    state_text=(
                        self._t(state_key)
                        if state_key != reading.state
                        else reading.state
                    ),
                    stroke=stroke,
                    wash=wash,
                    ink=_INK,
                    ink_faint=_INK_FAINT,
                    rule=_RULE,
                    surface=_SURFACE,
                    ground=_GROUND,
                    swatch=_swatch(reading.color, 22)
                    if reading.color
                    else None,
                    retention_floor=report.minimum_margin_retention,
                )
            )
        self._rows_layout.addStretch(1)

        # First sentence on screen, the rest on hover. An operator reading
        # this at shift start needs the next action, not the reasoning.
        remedy = self._remedy_text(report)
        self._remedy_label.setText(_first_sentence(remedy))
        self._remedy_label.setToolTip(remedy)
        verdict_key = _STATUS_VERDICT_KEYS.get(report.status)
        self._show_verdict(
            report.status,
            self._t(verdict_key)
            if verdict_key
            else self._t("preflight_verdict", status=report.status),
        )
        # Only a board that read correctly may become the bar for later shifts,
        # and the store enforces that too -- this just stops the button
        # inviting a refusal. A baseline that predates the current contract is
        # deliberately *not* a bar to recording: the reference is bound to that
        # baseline's hash, so it expires when the baseline is rebuilt, and a
        # station can watch its own drift in the meantime.
        self._record_btn.setEnabled(report.status in {STATUS_OK, STATUS_WARN})

    def _show_verdict(self, status: str, text: str) -> None:
        stroke, wash = _STATUS_COLORS.get(status, (_INK, _SURFACE))
        self._verdict_label.setText(text)
        self._verdict_label.setStyleSheet(
            f"background-color: {wash}; color: {stroke};"
            f" border: 1px solid {stroke};"
        )
        self._remedy_label.setVisible(status in {STATUS_WARN, STATUS_NG})

    def _remedy_text(self, report: ColorPreflightReport) -> str:
        """Name the step that fits *this* reading.

        A fixed remedy sent an operator to check the lighting when the lighting
        was fine and the baseline was stale, which is how advice stops being
        read at all.

        The baseline note is appended rather than competing for the one slot:
        when there is also no reference yet, the actionable step is to record
        one, and an operator who saw only a baseline warning would reasonably
        wonder whether recording was safe. It is.
        """
        return "\n".join(
            self._t(_REMEDY_KEYS[cause]) for cause in remedy_causes(report)
        )

    def _append_to_ledger(
        self, report: ColorPreflightReport, source: ColorPreflightSource
    ) -> None:
        """Every run is evidence, pass or fail; a failed write is not fatal.

        Losing the log line would be a worse outcome as a blocked dialog than as
        a message: the operator still needs the verdict on screen.
        """
        try:
            self._ledger.append(
                ColorPreflightRecord(
                    recorded_at=source.taken_at,
                    operator="",
                    product=self._product,
                    area=self._area,
                    model_type=self._model_type,
                    model_version=source.model_version,
                    report=report,
                )
            )
        except ColorPreflightStoreError as exc:
            QMessageBox.warning(
                self,
                self._t("preflight_title"),
                self._t("preflight_ledger_failed", error=exc),
            )

    # ------------------------------------------------------------------
    def _poll_for_new_inspection(self) -> None:
        """Re-read when a newer inspection lands, and only then.

        Re-reading on every tick would append a ledger line every few seconds
        and turn the trend into noise.
        """
        try:
            newest = latest_inspection_snapshot(
                self._results_root,
                self._product,
                self._area,
                self._model_type,
            )
        except OSError:
            return
        if newest is None or newest == self._seen_snapshot:
            return
        self.refresh()

    def reject(self) -> None:  # noqa: D102 - stop the watcher with the dialog
        timer = getattr(self, "_watch_timer", None)
        if timer is not None:
            timer.stop()
        super().reject()

    # ------------------------------------------------------------------
    def _ask_operator(self) -> tuple[str, bool]:
        name, accepted = QInputDialog.getText(
            self,
            self._t("preflight_operator_title"),
            self._t("preflight_operator_prompt"),
        )
        return str(name or "").strip(), bool(accepted)

    def _on_record(self) -> None:
        if self._report is None:
            return
        operator, accepted = self._operator_prompt_fn()
        if not accepted:
            return
        try:
            recorded = self._record_fn(
                self._config_path,
                self._report,
                operator=operator,
                baseline_sha256=self._baseline_sha256,
            )
        except ColorPreflightStoreError as exc:
            QMessageBox.warning(
                self,
                self._t("preflight_title"),
                self._t("preflight_record_failed", error=exc),
            )
            return
        QMessageBox.information(
            self,
            self._t("preflight_title"),
            self._t(
                "preflight_recorded",
                operator=getattr(recorded, "recorded_by", operator),
                when=getattr(recorded, "recorded_at", ""),
            ),
        )
        # The reference the next comparison uses has changed, so what is on
        # screen is now judged against something else.
        self.refresh()


def _format_retention(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"
