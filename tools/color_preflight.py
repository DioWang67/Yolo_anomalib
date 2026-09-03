"""Pre-shift color check over a reference-board inspection.

The operator ritual this belongs to already exists: put the golden sample in
the fixture, run the illumination calibration until mean luma is back inside
tolerance. That closes the loop on brightness. This closes it on color -- it
reads one inspection of that same board and reports whether every expected
color still reads as itself, and how much threshold headroom is left compared
with the day the reference was recorded.

The same check is in the GUI under 燈光控制 > 顏色開線檢查, over the same
service; this command is the engineer's way in, and the only way to see the
trend across shifts.

It never writes a baseline. ``--record-reference`` writes only the reference
margins, and only from a clean board, under a name.

Usage:
    python tools/color_preflight.py --product Cable1 --area A
    python tools/color_preflight.py --product Cable1 --area A --trend 14
    python tools/color_preflight.py --product Cable1 --area A \
        --record-reference --operator "line-lead-a"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.services.color_preflight import (
    REMEDY_BASELINE,
    REMEDY_MARGIN,
    REMEDY_MISREAD,
    REMEDY_NO_REFERENCE,
    REMEDY_STALE_REFERENCE,
    STATE_OK,
    STATUS_NG,
    STATUS_OK,
    STATUS_WARN,
    ColorPreflightReport,
    remedy_causes,
)
from core.services.color_preflight_runner import (
    ColorPreflightSource,
    ColorPreflightUnavailable,
    run_color_preflight,
)
from core.services.color_preflight_store import (
    ColorPreflightLedger,
    ColorPreflightRecord,
    ColorPreflightStoreError,
    record_reference_margins,
)
from core.services.station_color_settings import station_config_path
from core.station_data import load_station_data_paths

EXIT_OK = 0
EXIT_NG = 1
EXIT_USAGE = 2

_STATUS_LABELS = {STATUS_OK: "OK", STATUS_WARN: "WARN", STATUS_NG: "NG"}

#: The shared remedy causes as this front end words them. Which causes apply is
#: decided once in ``core.services.color_preflight``, so the dialog and this
#: command cannot end up giving different advice about the same reading.
_REMEDY_TEXT = {
    REMEDY_MISREAD: (
        "先檢查治具與光源，再做光源校正。仍未恢復則升級為顏色基準重建 + "
        "重跑驗收 + 簽核。不要用這次讀數去改基準。"
    ),
    REMEDY_MARGIN: (
        "顏色都還判對，但餘裕在衰退 —— 通常是光源老化。先做光源校正；"
        "若連續幾班持續下滑，就要規劃基準重建。"
    ),
    REMEDY_STALE_REFERENCE: (
        "過期的只是參考餘裕，不是這片板子。確認樣本與光源良好後，用 "
        "--record-reference 重新記錄一次。"
    ),
    REMEDY_NO_REFERENCE: (
        "目前沒有問題，只是沒有可比對的基準。若樣本與光源確認良好，用 "
        "--record-reference --operator <姓名> 把這片板子記錄為參考。"
    ),
    REMEDY_BASELINE: (
        "另外：已部署基準早於現行契約（見上面的「基準來歷」）。那要靠基準重建 "
        "+ 驗收 + 簽核處理，不是開線時能解決的；在那之前漂移偵測仍可用，"
        "參考餘裕會綁定目前這份基準，重建後自動失效。"
    ),
}


def _format_margin(value: float | None) -> str:
    return "     -" if value is None else f"{value:+.3f}"


def _format_retention(value: float | None) -> str:
    return "    -" if value is None else f"{value * 100:4.0f}%"


def _print_report(
    report: ColorPreflightReport, source: ColorPreflightSource
) -> None:
    age = source.age
    print(f"快照     : {source.snapshot_path}")
    print(f"拍攝時間 : {source.taken_at}" + (f"（{age}）" if age else ""))
    print(f"保留率門檻: {report.minimum_margin_retention:.2f}")
    if report.reference_is_stale:
        print("參考餘裕 : 已過期 —— 是對著另一份基準檔量的，數字不可比")
    elif report.reference_is_missing:
        print("參考餘裕 : 尚未建立 —— 這次只能回報量到的值")
    if report.baseline_provenance_failure:
        print(f"基準來歷 : {report.baseline_provenance_failure}")
    print()
    print(f"{'色別':<10}{'數量':>7}  {'餘裕':>7}  {'參考':>7}  {'保留':>6}  狀態")
    print("-" * 58)
    for item in report.colors:
        counts = f"{item.observed_count}/{item.expected_count}"
        name = item.color or "(未量測)"
        flag = "" if item.state == STATE_OK else "  <-"
        print(
            f"{name:<10}{counts:>7}  "
            f"{_format_margin(item.margin):>7}  "
            f"{_format_margin(item.reference_margin):>7}  "
            f"{_format_retention(item.retention):>6}  "
            f"{item.state}{flag}"
        )
    print("-" * 58)
    print(f"判定     : {_STATUS_LABELS.get(report.status, report.status)}")
    if report.status != STATUS_OK:
        for index, cause in enumerate(remedy_causes(report)):
            label = "處置     : " if index == 0 else "           "
            print(label + _REMEDY_TEXT[cause])


def _print_trend(
    ledger: ColorPreflightLedger,
    product: str,
    area: str,
    model_type: str,
    limit: int,
) -> None:
    records = ledger.read(product, area, model_type, limit=limit)
    if not records:
        print("尚無開線紀錄。")
        return
    colors: list[str] = []
    for record in records:
        for item in record.get("colors") or []:
            name = str(item.get("color") or "")
            if name and name not in colors:
                colors.append(name)
    print(f"{'時間':<21}{'判定':<6}" + "".join(f"{name:>9}" for name in colors))
    print("-" * (27 + 9 * len(colors)))
    for record in records:
        margins = {
            str(item.get("color") or ""): item.get("margin")
            for item in record.get("colors") or []
        }
        row = "".join(f"{_format_margin(margins.get(name)):>9}" for name in colors)
        stamp = str(record.get("recorded_at") or "")[:19]
        print(f"{stamp:<21}{str(record.get('status') or ''):<6}{row}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--model-type", default="yolo")
    parser.add_argument(
        "--snapshot", help="評估指定的檢測快照，而不是該工位最新的一筆"
    )
    parser.add_argument(
        "--trend",
        type=int,
        metavar="N",
        help="只列出最近 N 筆開線紀錄的餘裕趨勢，不做新的評估",
    )
    parser.add_argument(
        "--record-reference",
        action="store_true",
        help="把這次的餘裕記錄為之後每班比對的參考（需 --operator）",
    )
    parser.add_argument("--operator", default="", help="記錄參考餘裕的具名操作者")
    parser.add_argument(
        "--no-ledger", action="store_true", help="不要把這次評估寫進開線紀錄"
    )
    args = parser.parse_args(argv)

    paths = load_station_data_paths()
    ledger = ColorPreflightLedger(paths.color_preflight)

    if args.trend is not None:
        _print_trend(ledger, args.product, args.area, args.model_type, args.trend)
        return EXIT_OK

    config_path = station_config_path(
        paths.models, args.product, args.area, args.model_type
    )
    try:
        report, source = run_color_preflight(
            config_path=config_path,
            results_root=paths.results,
            product=args.product,
            area=args.area,
            model_type=args.model_type,
            snapshot_path=args.snapshot,
        )
    except ColorPreflightUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE

    _print_report(report, source)

    if not args.no_ledger:
        try:
            written = ledger.append(
                ColorPreflightRecord(
                    recorded_at=source.taken_at,
                    operator=args.operator,
                    product=args.product,
                    area=args.area,
                    model_type=args.model_type,
                    model_version=source.model_version,
                    report=report,
                )
            )
        except ColorPreflightStoreError as exc:
            print(f"開線紀錄寫入失敗：{exc}", file=sys.stderr)
            return EXIT_NG
        print(f"紀錄     : {written}")

    if args.record_reference:
        try:
            recorded = record_reference_margins(
                config_path,
                report,
                operator=args.operator,
                baseline_sha256=source.baseline_sha256,
            )
        except ColorPreflightStoreError as exc:
            print(f"\n無法建立參考餘裕：{exc}", file=sys.stderr)
            return EXIT_NG
        print(
            f"\n已建立參考餘裕（{recorded.recorded_by} / {recorded.recorded_at}）："
            + "、".join(
                f"{name} {value:+.3f}"
                for name, value in sorted(recorded.reference_margins.items())
            )
        )

    return EXIT_NG if report.status == STATUS_NG else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
