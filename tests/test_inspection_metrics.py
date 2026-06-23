import csv

from tools.inspection_metrics import (
    build_confusion_matrix,
    compute_report,
    load_manifest_rows,
    metrics_from_matrix,
)


def _rows(*specs: tuple[str, str, str, str]) -> list[dict[str, str]]:
    """Build manifest rows from (product, area, status, review_label) tuples."""
    return [
        {"product": p, "area": a, "status": s, "review_label": label}
        for p, a, s, label in specs
    ]


def test_confusion_matrix_maps_labels_to_cells():
    rows = _rows(
        ("PCBA1", "A", "FAIL", "confirmed_ng"),      # TP
        ("PCBA1", "A", "DETECTION_FAIL", "false_positive"),  # FP (過殺)
        ("PCBA1", "A", "PASS", "false_negative"),    # FN (漏檢)
        ("PCBA1", "A", "PASS", "confirmed_ok"),      # TN
        ("PCBA1", "A", "FAIL", "uncertain"),         # excluded
        ("PCBA1", "A", "FAIL", ""),                  # unlabeled
    )
    matrix = build_confusion_matrix(rows)
    assert (matrix.tp, matrix.fp, matrix.fn, matrix.tn) == (1, 1, 1, 1)
    assert matrix.uncertain == 1
    assert matrix.unlabeled == 1
    assert matrix.inconsistent == 0


def test_escape_recall_precision_math():
    # TP=8, FN=2 -> recall 0.8, escape 0.2 ; FP=2 -> precision 0.8
    rows = (
        _rows(*[("P", "A", "FAIL", "confirmed_ng")] * 8)
        + _rows(*[("P", "A", "PASS", "false_negative")] * 2)
        + _rows(*[("P", "A", "FAIL", "false_positive")] * 2)
    )
    metrics = metrics_from_matrix(build_confusion_matrix(rows), scope="overall")
    assert metrics.recall == 0.8
    assert metrics.escape_rate == 0.2
    assert metrics.precision == 0.8


def test_inconsistent_label_vs_status_is_flagged():
    # confirmed_ng implies machine flagged NG, but status is PASS -> inconsistent
    matrix = build_confusion_matrix(_rows(("P", "A", "PASS", "confirmed_ng")))
    assert matrix.inconsistent == 1
    assert matrix.tp == 1  # still classified by label


def test_escape_rate_undefined_without_ng_cases():
    metrics = metrics_from_matrix(
        build_confusion_matrix(_rows(("P", "A", "FAIL", "false_positive"))),
        scope="overall",
    )
    assert metrics.escape_rate is None
    assert metrics.recall is None
    assert any("PASS cases" in note for note in metrics.notes)


def test_per_station_grouping():
    rows = _rows(
        ("PCBA1", "A", "FAIL", "confirmed_ng"),
        ("PCBA1", "B", "PASS", "false_negative"),
    )
    report = compute_report(rows)
    assert report["total_rows"] == 2
    scopes = {block["scope"] for block in report["per_station"]}
    assert scopes == {"PCBA1/A", "PCBA1/B"}


def test_load_manifest_rows_roundtrip(tmp_path):
    path = tmp_path / "review_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["product", "area", "status", "review_label"]
        )
        writer.writeheader()
        writer.writerow(
            {"product": "P", "area": "A", "status": "FAIL", "review_label": "confirmed_ng"}
        )
    rows = load_manifest_rows(path)
    report = compute_report(rows)
    assert report["overall"]["matrix"]["tp"] == 1
