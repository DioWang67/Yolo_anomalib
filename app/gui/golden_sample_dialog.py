"""Golden-board start-of-shift workflow; all image I/O runs in one worker."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, SimpleQueue
from uuid import uuid4

import cv2
from PyQt5.QtCore import QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
)

from app.gui.golden_sample_view import GoldenEvidenceView, color_label, position_label
from core.services.color_preflight_runner import ColorPreflightUnavailable, read_station_config, resolve_color_model
from core.services.golden_sample import (
    BASELINE_COUNT,
    CHECK_COUNT,
    GoldenSampleError,
    color_identity_payload,
    condition_drift,
    configuration_identity,
    evaluate_readings,
    finalize_baseline,
    margin_retention,
    measure_baseline,
    normal_delta_e_samples,
    observed_conditions,
    propose_limits,
    measure_snapshot,
    read_json,
    validate_reference,
    write_json_atomic,
)
from core.services.golden_sample_preview import preview_or_empty
from core.services.station_color_settings import StationColorSettingsError, station_expected_color_positions

#: How long one triggered inspection may take to publish complete evidence.
CAPTURE_TIMEOUT_SECONDS = 120
#: Slack on top of ``required x CAPTURE_TIMEOUT_SECONDS`` for the whole session.
COLLECTION_GRACE_SECONDS = 120
#: Re-reads of the same snapshot while its crops are still being written.
EVIDENCE_READ_ATTEMPTS = 3
#: Shots that may be discarded and retaken before the session gives up.
CAPTURE_RETRY_BUDGET = 2
#: Spin-box sentinel meaning "no limit yet"; shown as 待量測後提議.
UNSET_LIMIT = 0.01
#: Advisory only -- an old baseline still judges, it just says how old it is.
BASELINE_AGE_HINT_DAYS = 30


def local_time(value) -> str:
    """A stored UTC timestamp as local wall-clock time, to the minute.

    Operators read this to decide whether a baseline is today's. Microseconds
    and a ``+00:00`` offset made that judgement require mental arithmetic.
    """
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return "?"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def baseline_age_days(value) -> float | None:
    try:
        moment = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - moment).total_seconds() / 86400


def station_identity(config_path, observed_override=None):
    """``(identity, retention, identity_payload, conditions)`` for a station.

    One reader for both the worker and the dialog: they must never disagree
    about whether the stored baseline still applies.

    ``observed_override`` supplies camera values in force that the config file
    does not record; identity and retention are unaffected by it.
    """
    config = read_station_config(config_path)
    model = resolve_color_model(config_path, str(config.get("color_model_path") or ""))
    return (
        configuration_identity(config_path, model, config=config),
        margin_retention(config),
        color_identity_payload(config),
        observed_conditions(config, observed_override),
    )


class GoldenCaptureWorker(QThread):
    capture_requested = pyqtSignal()
    progress = pyqtSignal(int, int, str)
    proposal_ready = pyqtSignal(object)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    preview_ready = pyqtSignal(object)

    def __init__(
        self,
        *,
        config_path,
        results_root,
        scope,
        reference_path,
        baseline,
        operator,
        sample_id,
        delta_e,
        repeatability,
        automatic=False,
        observed_override_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self.observed_override_fn = observed_override_fn
        self.config_path = config_path
        self.results_root = results_root
        self.scope = scope
        self.reference_path = reference_path
        self.baseline = baseline
        self.operator = operator
        self.sample_id = sample_id
        self.delta_e = delta_e
        self.repeatability = repeatability
        self.automatic = automatic
        self.capture_starts = SimpleQueue()
        self.started_at = time.time()

    def run(self):
        try:
            self._collect()
        except (
            GoldenSampleError,
            ColorPreflightUnavailable,
            StationColorSettingsError,
            OSError,
            cv2.error,
            UnicodeError,
        ) as exc:
            self.failed.emit(str(exc))

    def _identity(self):
        return station_identity(self.config_path)[0]

    def _history(self):
        """Worst delta-E from this station's past checks, for the proposal."""
        payloads = []
        for path in sorted(self.reference_path.parent.glob("check-*.json")):
            try:
                payloads.append(read_json(path))
            except GoldenSampleError:
                continue
        return normal_delta_e_samples(payloads)

    def _report(self, collected, required, message):
        """Single source of progress: the dialog renders this and nothing else.

        Two independently maintained counters (shots triggered vs evidence
        accepted) used to be on screen at once, disagreeing during every
        retry.
        """
        self.progress.emit(collected, required, message)

    def _observed_override(self):
        """Live camera values the config file does not carry, or {}."""
        if self.observed_override_fn is None:
            return {}
        try:
            return self.observed_override_fn() or {}
        except Exception:  # noqa: BLE001 - reporting must not fail a check
            return {}

    def _collect(self):
        identity, retention, identity_payload, conditions = station_identity(
            self.config_path, self._observed_override()
        )
        expected = station_expected_color_positions(self.config_path, *self.scope[:2])
        if not expected:
            raise GoldenSampleError("站點未設定預期顏色，無法檢查 golden sample")
        reference = None
        if not self.baseline:
            reference = read_json(self.reference_path)
            validate_reference(reference, identity)
            if self.sample_id.strip() != reference["sample_id"]:
                raise GoldenSampleError("golden sample 編號與正常基準不一致")
        required = BASELINE_COUNT if self.baseline else CHECK_COUNT
        # Every shot may legitimately need its full per-capture budget, so a
        # flat total used to make a slow station fail the fifth baseline
        # capture by arithmetic alone (5 x 120 == the old 600s ceiling).
        # Compared against the live ``started_at`` rather than a precomputed
        # instant, so a restarted clock is honoured.
        budget = required * CAPTURE_TIMEOUT_SECONDS + COLLECTION_GRACE_SECONDS
        readings, seen, retries = [], set(), {}
        discarded = 0
        product, area, model_type = self.scope
        pattern = f"*/{product}/{area}/*/metadata/{model_type}/*_config_snapshot.json"
        last_reading_at = time.monotonic()
        capture_started_at = None
        if self.automatic:
            self.capture_requested.emit()
        self._report(0, required, f"準備收集 {required} 次檢測，請保持樣品不動")
        while not self.isInterruptionRequested():
            if self.automatic and time.monotonic() - last_reading_at > CAPTURE_TIMEOUT_SECONDS:
                raise GoldenSampleError(
                    f"第 {len(readings) + 1} 次檢測超過 {CAPTURE_TIMEOUT_SECONDS} 秒仍未收到完整結果，"
                    "請檢查相機與結果儲存設定"
                )
            if time.time() - self.started_at > budget:
                raise GoldenSampleError(f"收集超過 {int(budget / 60)} 分鐘，請重新開始本次檢查")
            if self.automatic and capture_started_at is None:
                try:
                    capture_started_at = self.capture_starts.get_nowait()
                except Empty:
                    self.msleep(100)
                    continue
            candidates = sorted(self.results_root.glob(pattern), key=lambda p: p.stat().st_mtime)
            for path in candidates:
                if self.isInterruptionRequested():
                    return
                if path in seen or path.stat().st_mtime < self.started_at:
                    continue
                try:
                    payload = read_json(path)
                    taken = datetime.fromisoformat(str(payload.get("timestamp") or ""))
                    if taken.timestamp() < (capture_started_at if self.automatic else self.started_at):
                        seen.add(path)
                        continue
                    if taken.timestamp() > time.time() + 5:
                        raise GoldenSampleError("檢測時間在未來，請確認站點時鐘")
                    reading = measure_snapshot(path, expected)
                except (GoldenSampleError, ValueError, OSError) as exc:
                    # Two different failures wear the same exception here.
                    # Snapshot and crops are published as separate filesystem
                    # writes, so an incomplete read is usually just early --
                    # re-read the same file. Once it has stopped changing, the
                    # shot itself is bad: drop that shot and take another,
                    # keeping the evidence already collected. Throwing the
                    # whole session away made a single bad frame cost an
                    # operator all five baseline captures.
                    retries[path] = retries.get(path, 0) + 1
                    if retries[path] < EVIDENCE_READ_ATTEMPTS:
                        self._report(len(readings), required, "等待本次影像完整寫入…")
                        break
                    seen.add(path)
                    discarded += 1
                    if discarded > CAPTURE_RETRY_BUDGET:
                        raise GoldenSampleError(
                            f"連續 {discarded} 次證據不完整，已停止收集：{exc}"
                        ) from exc
                    self._report(
                        len(readings),
                        required,
                        f"第 {len(readings) + 1} 次證據不完整，"
                        + ("重拍中" if self.automatic else "請再執行一次檢測")
                        + f"（第 {discarded} 次重試）：{exc}",
                    )
                    if self.automatic:
                        capture_started_at = None
                        last_reading_at = time.monotonic()
                        self.capture_requested.emit()
                    break
                if station_identity(self.config_path)[0] != identity:
                    raise GoldenSampleError("收集期間設定或顏色模型變更，請重新開始")
                seen.add(path)
                readings.append(reading)
                last_reading_at = time.monotonic()
                self._report(len(readings), required, f"已收集 {len(readings)} / {required} 次新檢測")
                if len(readings) < required:
                    if self.automatic:
                        capture_started_at = None
                        self.capture_requested.emit()
                        break
                    continue
                if self.isInterruptionRequested():
                    return
                if self.baseline:
                    # Measure, propose, and stop. Nothing is stored until an
                    # engineer has seen what this station actually does and
                    # accepted the limits it will be judged against.
                    measured = measure_baseline(
                        readings,
                        identity=identity,
                        operator=self.operator,
                        sample_id=self.sample_id,
                    )
                    self.proposal_ready.emit(
                        {
                            "measured": measured,
                            "proposal": propose_limits(
                                measured["measured_jitter"], self._history()
                            ),
                            "retention": retention,
                            "identity_payload": identity_payload,
                            "conditions": conditions,
                        }
                    )
                    return
                else:
                    result = evaluate_readings(readings, reference, identity)
                    result["observed_conditions"] = conditions
                    # Visible, but never a verdict: a moved exposure is what
                    # the delta-E measurement above is there to judge.
                    result["condition_drift"] = condition_drift(
                        reference.get("observed_conditions"), conditions
                    )
                    write_json_atomic(self.reference_path.parent / f"check-{uuid4().hex}.json", result)
                self.preview_ready.emit(
                    preview_or_empty(result if self.baseline else reference, None if self.baseline else result)
                )
                self.completed.emit(result)
                return
            # Sleep in short slices so closing never destroys a running thread.
            for _ in range(20):
                if self.isInterruptionRequested():
                    return
                self.msleep(100)


class GoldenPreviewWorker(QThread):
    ready = pyqtSignal(object)

    def __init__(self, reference, parent=None):
        super().__init__(parent)
        self.reference = reference

    def run(self):
        previews = preview_or_empty(self.reference)
        if not self.isInterruptionRequested():
            self.ready.emit(previews)


class GoldenSampleDialog(QDialog):
    def __init__(
        self,
        *,
        config_path,
        results_root,
        product,
        area,
        model_type,
        ledger,
        language="en",
        request_capture_fn=None,
        readiness_fn=None,
        authorize_fn=None,
        observed_override_fn=None,
        parent=None,
    ):
        super().__init__(parent)
        self._observed_override_fn = observed_override_fn
        self._config_path = Path(config_path)
        self._results_root = Path(results_root)
        self._scope = (product, area, model_type)
        self._reference_path = ledger.path_for(*self._scope).with_suffix("") / "golden-reference.json"
        self._worker = None
        self._preview_workers = []
        self._reference_stamp = None
        self._reference_loaded = False
        self._preview_generation = 0
        self._required_count = 0
        self._cancelled = False
        self._pending = None
        self._proposed = None
        self._request_capture_fn = request_capture_fn
        self._readiness_fn = readiness_fn
        self._authorize_fn = authorize_fn
        self._ready = readiness_fn is None
        self._capture_timer = QTimer(self)
        self._capture_timer.setInterval(200)
        self._capture_timer.timeout.connect(self._request_capture)
        # Preconditions used to surface only after the operator pressed start
        # and waited for the abort. Polling keeps the answer on screen before
        # they commit; it only reads GUI-thread-owned widget state.
        self._readiness_timer = QTimer(self)
        self._readiness_timer.setInterval(500)
        self._readiness_timer.timeout.connect(self._refresh_readiness)
        self.setWindowTitle("Golden sample 顏色穩定性檢查")
        self.resize(1080, 850)
        self.setStyleSheet(
            "QDialog {background: #f3f6fa;} QPushButton {padding: 9px 14px;}"
            "QPushButton#startGolden {background: #176c4b; color: white; font-weight: bold;"
            " border-radius: 5px; font-size: 17px; padding: 14px 18px;}"
            "QPushButton#startGolden:disabled {background: #acbab3;}"
            "QCheckBox#confirmSample {font-size: 15px; padding: 10px; background: white;"
            " border: 1px solid #c6d3e0; border-radius: 5px;}"
        )
        layout = QVBoxLayout(self)
        self._verdict = QLabel("待檢查 — 尚未確認本次開線條件")
        self._style_verdict("waiting")
        self._verdict.setWordWrap(True)
        layout.addWidget(self._verdict)
        self._reference_label = QLabel()
        self._reference_label.setWordWrap(True)
        self._reference_label.setStyleSheet("color: #42556b;")
        layout.addWidget(self._reference_label)
        self._readiness_label = QLabel()
        self._readiness_label.setWordWrap(True)
        self._readiness_label.setVisible(readiness_fn is not None)
        layout.addWidget(self._readiness_label)
        self._confirm = QCheckBox("我已放妥 golden sample，且收集期間不更換樣品／設定")
        self._confirm.setObjectName("confirmSample")
        layout.addWidget(self._confirm)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, CHECK_COUNT)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("已收集 %v / %m 次")
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        self._views = QTabWidget()
        self._evidence = GoldenEvidenceView()
        self._views.addTab(self._evidence, "影像對照")
        self._table = QTableWidget(0, 7)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setHorizontalHeaderLabels(
            ["位置／顏色", "最大 ΔE76", "平均 ΔL*", "連拍波動", "辨識餘裕", "結果", "原因"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        # The first column already carries the position number.
        self._table.verticalHeader().setVisible(False)
        self._views.addTab(self._table, "數值明細")
        layout.addWidget(self._views, 1)
        legend = QLabel(
            "青框＝固定取樣區域　｜　4×4 數字＝各區域跨次最大色差；紅底表示超限。基準圖片為實拍範例，判定使用多張統計。"
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        self._maintenance_toggle = QPushButton("工程維護：基準與容許範圍（需工程 PIN）")
        self._maintenance_toggle.setCheckable(True)
        layout.addWidget(self._maintenance_toggle)
        maintenance = self._maintenance = QGroupBox("建立／更換正常基準（5 次新檢測，舊基準會封存）")
        # Authorisation is decided before the panel opens, so the toggle routes
        # through a guard rather than driving visibility directly.
        self._maintenance_toggle.toggled.connect(self._toggle_maintenance)
        maintenance.setVisible(False)
        form = QFormLayout(maintenance)
        self._operator = QLineEdit()
        self._sample_id = QLineEdit()
        self._sample_id.setPlaceholderText("golden sample 編號")
        self._delta = QDoubleSpinBox()
        self._repeatability = QDoubleSpinBox()
        for field in (self._delta, self._repeatability):
            # Empty until this station has been measured. A hard-coded 3.00/1.00
            # reads as a recommendation it is not, and asking the engineer to
            # invent a number before any measurement exists is no better -- so
            # the limits are proposed from the measurement, then confirmed.
            field.setRange(UNSET_LIMIT, 100)
            field.setDecimals(2)
            field.setSpecialValueText("待量測後提議")
            field.setValue(UNSET_LIMIT)
        form.addRow("建立人員", self._operator)
        form.addRow("golden sample 編號", self._sample_id)
        form.addRow("最大區域色差上限 ΔE76", self._delta)
        form.addRow("連拍波動上限 ΔE76", self._repeatability)
        self._measured_hint = QLabel()
        self._measured_hint.setWordWrap(True)
        self._measured_hint.setStyleSheet("color: #42556b;")
        form.addRow(self._measured_hint)
        self._engineering = QCheckBox("已確認環境正常（上限將依實測提議，可自行調整）")
        form.addRow(self._engineering)
        self._baseline_button = QPushButton("① 收集 5 次並量測")
        self._baseline_button.clicked.connect(lambda: self._start(True))
        form.addRow(self._baseline_button)
        self._save_button = QPushButton("② 接受上限並儲存正常基準")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._save_baseline)
        form.addRow(self._save_button)
        layout.addWidget(maintenance)
        buttons = QHBoxLayout()
        self._start_button = QPushButton("開始開線檢查（3 次）")
        self._start_button.setObjectName("startGolden")
        self._start_button.clicked.connect(lambda: self._start(False))
        self._cancel_button = QPushButton("取消本次收集")
        self._cancel_button.clicked.connect(self._cancel)
        self._cancel_button.setEnabled(False)
        close = QPushButton("回主畫面（保留進度）")
        close.clicked.connect(self.close)
        for button in (self._start_button, self._cancel_button, close):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._load_reference_label()

    def _load_reference_label(self):
        if not self._reference_path.exists():
            self._reference_loaded = False
            self._update_start_enabled()
            # Deliberately not auto-opening the panel: it now sits behind the
            # engineering PIN, and prompting for one just by opening the window
            # would train operators to fetch an engineer for a daily check.
            self._reference_label.setText(
                "尚無 golden sample 正常基準；請由工程人員在「工程維護」建立。"
                "舊的分數餘裕不會自動轉為正常基準。"
            )
            return
        try:
            reference = read_json(self._reference_path)
            identity = station_identity(self._config_path)[0]
            validate_reference(reference, identity)
            self._reference_loaded = True
            self._reference_stamp = self._reference_path.stat().st_mtime_ns
            self._sample_id.setText(reference["sample_id"])
            self._delta.setValue(reference["delta_e_limit"])
            self._repeatability.setValue(reference["repeatability_limit"])
            self._operator.setText(reference.get("operator", ""))
            # The sample id belongs on the control the operator must actually
            # touch, instead of being repeated across three widgets.
            self._confirm.setText(f"我已放妥標準品 {reference['sample_id']}，取樣期間保持不動")
            self._maintenance_toggle.setChecked(False)
            self._measured_hint.setText(
                f"目前基準建立時實測最大波動 {reference.get('measured_jitter', 0.0):.2f}"
                f"（當時設定的波動上限 {reference.get('repeatability_limit', 0):.2f}）"
            )
            age = baseline_age_days(reference.get("created_at"))
            stale = "" if age is None or age < BASELINE_AGE_HINT_DAYS else f"　⚠ 已使用 {int(age)} 天"
            self._reference_label.setText(
                f"基準樣品 {reference.get('sample_id', '?')}"
                f"　建立於 {local_time(reference.get('created_at'))}{stale}"
                f"　｜　色差上限 {reference.get('delta_e_limit', '?')}"
                f"、波動上限 {reference.get('repeatability_limit', '?')}"
                f"、餘裕保留 {float(reference.get('margin_retention', 0)):.0%}"
            )
            self._refresh_readiness()
            self._preview_generation += 1
            generation = self._preview_generation
            worker = GoldenPreviewWorker(reference, self)
            self._preview_workers.append(worker)
            worker.ready.connect(
                lambda previews: self._evidence.show_previews(previews)
                if generation == self._preview_generation
                else None
            )
            worker.finished.connect(lambda: self._finish_preview(worker))
            worker.start()
        except (GoldenSampleError, ColorPreflightUnavailable, OSError, StationColorSettingsError) as exc:
            self._reference_loaded = False
            self._update_start_enabled()
            self._reference_label.setText(str(exc))

    def _finish_preview(self, worker):
        self._preview_workers.remove(worker)
        worker.deleteLater()

    def _update_start_enabled(self):
        self._start_button.setEnabled(self._reference_loaded and self._worker is None and self._ready)

    def _refresh_readiness(self):
        """Show the station preconditions before the operator commits to a run.

        The same checks the capture adapter enforces, so the panel and the
        actual refusal can never disagree.
        """
        if self._readiness_fn is None:
            # No host to ask (tests, or a dialog opened standalone): readiness
            # is assumed, but the start button must still be re-evaluated.
            self._update_start_enabled()
            return
        try:
            checks = list(self._readiness_fn())
        except Exception as exc:  # noqa: BLE001 - a broken host must not brick the dialog
            self._ready = False
            self._readiness_label.setText(f"無法確認站點狀態：{exc}")
            self._readiness_label.setStyleSheet("color: #8a4200;")
            self._update_start_enabled()
            return
        blocking = [check for check in checks if not check.ok]
        self._ready = not blocking
        if self._ready:
            self._readiness_label.setText("✓ 站點條件就緒：" + "、".join(c.label for c in checks))
            self._readiness_label.setStyleSheet("color: #155b3f;")
        else:
            self._readiness_label.setText(
                "尚未就緒 — " + "；".join(f"{c.label}：{c.hint}" for c in blocking)
            )
            self._readiness_label.setStyleSheet("color: #8a4200; font-weight: bold;")
        self._update_start_enabled()

    def _toggle_maintenance(self, expanded):
        """Engineering panel opens only behind the station's PIN."""
        if not expanded:
            self._maintenance.setVisible(False)
            return
        if self._authorize_fn is not None and not self._authorize_fn():
            # Revert without recursing: setChecked re-enters this slot.
            self._maintenance_toggle.blockSignals(True)
            self._maintenance_toggle.setChecked(False)
            self._maintenance_toggle.blockSignals(False)
            self._maintenance.setVisible(False)
            return
        self._maintenance.setVisible(True)

    def showEvent(self, event):
        super().showEvent(event)
        if self._readiness_fn is not None:
            self._refresh_readiness()
            self._readiness_timer.start()
        if self._worker is None:
            try:
                stamp = self._reference_path.stat().st_mtime_ns
            except OSError:
                stamp = None
            if stamp != self._reference_stamp:
                self._load_reference_label()

    def hideEvent(self, event):
        super().hideEvent(event)
        # Nothing to poll for while hidden; collection keeps running.
        self._readiness_timer.stop()

    def _start(self, baseline):
        if self._worker is not None:
            return
        if not self._confirm.isChecked() or not self._sample_id.text().strip():
            self._verdict.setText("請填寫樣品編號並確認 golden sample 已放妥")
            return
        if not baseline and not self._ready:
            self._verdict.setText("站點條件尚未就緒 — 請先排除上方列出的項目")
            return
        if baseline and (not self._engineering.isChecked() or not self._operator.text().strip()):
            self._verdict.setText("建立基準需要具名，並確認環境與容許上限已驗證")
            return
        self._cancelled = False
        if baseline:
            self._pending = None
            self._save_button.setEnabled(False)
        self._table.setRowCount(0)
        self._style_verdict("waiting")
        self._preview_generation += 1
        self._evidence.show_previews([])
        self._required_count = BASELINE_COUNT if baseline else CHECK_COUNT
        self._progress_bar.setRange(0, self._required_count)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._worker = GoldenCaptureWorker(
            config_path=self._config_path,
            results_root=self._results_root,
            scope=self._scope,
            reference_path=self._reference_path,
            baseline=baseline,
            operator=self._operator.text(),
            sample_id=self._sample_id.text(),
            delta_e=self._delta.value(),
            repeatability=self._repeatability.value(),
            automatic=self._request_capture_fn is not None,
            observed_override_fn=self._observed_override_fn,
            parent=self,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.proposal_ready.connect(self._on_proposal)
        self._worker.preview_ready.connect(self._evidence.show_previews)
        self._worker.capture_requested.connect(self._schedule_capture)
        self._worker.failed.connect(self._show_failure)
        self._worker.completed.connect(self._completed)
        self._worker.finished.connect(self._finished)
        self._set_busy(True)
        self._worker.start()

    def _on_proposal(self, payload):
        """Show what the station measured, and the limits that follow from it."""
        self._pending = payload
        proposal = payload["proposal"]
        self._delta.setValue(proposal["delta_e"])
        self._repeatability.setValue(proposal["repeatability"])
        self._proposed = (proposal["delta_e"], proposal["repeatability"])
        worst = [
            f"{position_label(index + 1, p['color'])} {p['jitter']:.2f}"
            for index, p in enumerate(payload["measured"]["positions"])
        ]
        self._measured_hint.setText(proposal["basis"] + "\n各位置實測波動：" + "、".join(worst))
        self._style_verdict("waiting")
        self._verdict.setText(
            f"量測完成 — 建議色差上限 {proposal['delta_e']:.2f}、波動上限 "
            f"{proposal['repeatability']:.2f}；確認後按「② 接受上限並儲存正常基準」"
        )
        self._maintenance_toggle.setChecked(True)
        self._save_button.setEnabled(True)

    def _save_baseline(self):
        """Store the measurement under the limits now on screen."""
        if self._pending is None:
            return
        delta, repeatability = self._delta.value(), self._repeatability.value()
        if min(delta, repeatability) <= UNSET_LIMIT:
            self._verdict.setText("請先確認色差與波動上限")
            return
        try:
            result = finalize_baseline(
                self._pending["measured"],
                delta_e=delta,
                repeatability=repeatability,
                retention=self._pending["retention"],
                identity_payload=self._pending["identity_payload"],
                conditions=self._pending["conditions"],
                limits_source=(
                    "proposed" if (delta, repeatability) == self._proposed else "adjusted"
                ),
            )
            # Keep every historical baseline, including the one being replaced.
            if self._reference_path.exists():
                write_json_atomic(
                    self._reference_path.parent / f"reference-{uuid4().hex}.json",
                    read_json(self._reference_path),
                )
            write_json_atomic(self._reference_path, result)
        except (GoldenSampleError, OSError) as exc:
            self._show_failure(str(exc))
            return
        self._pending = None
        self._save_button.setEnabled(False)
        self._engineering.setChecked(False)
        self._style_verdict("ok")
        self._verdict.setText(
            f"正常基準已儲存 — 色差上限 {delta:.2f}、波動上限 {repeatability:.2f}"
            f"（實測最大波動 {result['measured_jitter']:.2f}）；請另啟動一次開線檢查"
        )
        self._load_reference_label()

    def _on_progress(self, collected, required, message):
        # Progress is queued from the worker thread, so a signal emitted just
        # before cancellation can land after it. Letting it through would
        # replace "cancelled" with "preparing", which reads as still running.
        # Tracked here rather than via ``isInterruptionRequested``: Qt ignores
        # an interruption request on a thread that has not started yet, so the
        # worker's own flag can still be clear at this point.
        if self._worker is None or self._cancelled:
            return
        self._progress_bar.setRange(0, required)
        self._progress_bar.setValue(collected)
        self._verdict.setText(message)

    def _show_failure(self, message):
        self._style_verdict("attention")
        if "不穩定" in message:
            title = "取樣不穩定"
        elif any(word in message for word in ("位置", "裁切", "固定取樣")):
            title = "取樣位置／影像需確認"
        else:
            title = "檢查未完成"
        self._verdict.setText(f"{title} — {message}")

    def _schedule_capture(self):
        if self._worker is not None and not self._worker.isInterruptionRequested():
            self._capture_timer.start()

    def _request_capture(self):
        if self._worker is None or self._worker.isInterruptionRequested():
            self._capture_timer.stop()
            return
        # Stopped before the call, not after it. _request_capture_fn reaches
        # start_detection, which can raise a modal QMessageBox, and a modal
        # dialog spins its own event loop -- a timer still armed fires inside
        # that loop, re-enters here and stacks another dialog on the first.
        self._capture_timer.stop()
        try:
            # This callback runs on the GUI thread; the worker never touches UI.
            started_at = time.time()
            requested = self._request_capture_fn()
        except GoldenSampleError as exc:
            self.capture_failed(str(exc))
            return
        # The nested loop above can have finished or cancelled the run while
        # this call was in it, so the worker is re-read rather than assumed.
        worker = self._worker
        if worker is None or worker.isInterruptionRequested():
            return
        if requested:
            # Progress text stays the worker's to emit: it is the only place
            # that knows how much evidence was actually accepted.
            worker.capture_starts.put(started_at)
        else:
            # False means the previous pipeline is still releasing the camera;
            # the timer exists for exactly this retry, so it goes back on.
            self._capture_timer.start()

    def capture_failed(self, message):
        if self._worker is None:
            return
        self._capture_timer.stop()
        self._worker.requestInterruption()
        self._style_verdict("attention")
        self._verdict.setText("自動取樣中止 — " + message)

    def _set_busy(self, busy):
        for widget in (
            self._start_button,
            self._baseline_button,
            self._sample_id,
            self._confirm,
            self._engineering,
            self._operator,
            self._delta,
            self._repeatability,
            self._maintenance_toggle,
        ):
            widget.setEnabled(not busy)
        self._save_button.setEnabled(not busy and self._pending is not None)
        self._cancel_button.setEnabled(busy)
        if not busy:
            self._update_start_enabled()

    def _completed(self, result):
        if "rows" not in result:
            self._style_verdict("ok")
            self._verdict.setText(
                f"正常基準已建立 — 實測最大波動 {result.get('measured_jitter', 0.0):.2f}"
                f"（設定上限 {result.get('repeatability_limit', 0):.2f}）；請另啟動一次開線檢查"
            )
            self._load_reference_label()
            self._engineering.setChecked(False)
            return
        passed = result["status"] == "OK"
        self._style_verdict("ok" if passed else "attention")
        failed = [row for row in result["rows"] if row["reasons"]]
        # Conditions the calibration loop landed on are context for either
        # verdict, not a verdict of their own.
        drift = result.get("condition_drift") or []
        note = f"（相機條件已變動：{'、'.join(drift)}）" if drift else ""
        if passed:
            summary = "通過 — 本次顏色與波動符合已設定的正常範圍，可進行檢測" + note
        else:
            # Name the positions. "See the marked position below" was useless
            # when that card could be under the scroll fold.
            where = "、".join(position_label(row["position"], row["color"]) for row in failed)
            reasons = sorted({reason for row in failed for reason in row["reasons"]})
            summary = f"{'；'.join(reasons)} — {where}{note}"
        self._verdict.setText(summary)
        self._table.setRowCount(len(result["rows"]))
        for index, row in enumerate(result["rows"]):
            cells = [
                f"{row['position']} / {color_label(row['color'])}",
                f"{row['delta_e']:.2f}",
                f"{row['delta_l']:+.2f}",
                f"{row['jitter']:.2f}",
                f"{row['margin']:.3f}",
                "NG" if row["reasons"] else "OK",
                "；".join(row["reasons"]) or "穩定",
            ]
            for column, value in enumerate(cells):
                self._table.setItem(index, column, QTableWidgetItem(value))

    def _style_verdict(self, state):
        background, foreground = {"waiting": ("#e8eef6", "#243b53"),
                                  "ok": ("#e0f2e9", "#155b3f"),
                                  "attention": ("#fff0df", "#8a4200")}[state]
        self._verdict.setStyleSheet(f"font-size: 20px; font-weight: bold; padding: 12px;"
                                   f"border-radius: 6px; background: {background}; color: {foreground};")

    def _finished(self):
        self._capture_timer.stop()
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_busy(False)

    def _cancel(self):
        self._capture_timer.stop()
        if self._worker is not None:
            self._cancelled = True
            self._style_verdict("waiting")
            self._worker.requestInterruption()
            self._verdict.setText("本次已取消 — 未取得開線通過結果")

    def reject(self):
        self.close()

    def closeEvent(self, event):
        # Returning to production controls is part of collecting evidence.
        # Ignoring close prevents WA_DeleteOnClose from discarding the session.
        event.ignore()
        self.hide()

    def prepare_shutdown(self, timeout_ms: int = 2000) -> bool:
        """Stop collection before the owning main window destroys its children.

        A bounded wait avoids hanging the UI on slow filesystem I/O. The owner
        must defer destruction if the worker has not yet acknowledged cancel.
        """
        workers = list(self._preview_workers)
        if self._worker is not None:
            self._cancel()
            workers.append(self._worker)
        for worker in workers:
            worker.requestInterruption()
        deadline = time.monotonic() + timeout_ms / 1000
        return all(worker.wait(max(0, int((deadline - time.monotonic()) * 1000))) for worker in workers)
