"""Shared rendering for acceptance metrics shown in review tables."""

from __future__ import annotations


def format_count_with_rate(
    count: object,
    rate: object,
    unknown_reason: str,
) -> str:
    """Render a count and the rate derived from that same count, in one cell.

    Splitting the pair across neighbouring columns is what made the acceptance
    tables unreadable: a whole-verdict count sat beside a color-only rate, and
    no denominator a reader could guess turned the one into the other. Keeping
    both halves of one metric in one cell makes the pairing unambiguous.

    A ``None`` rate means the denominator is empty, and then the count carries
    no information either -- it is necessarily 0, and "0 escapes out of 0
    verifiable samples" is not a clean sheet. Both halves are replaced by
    ``unknown_reason`` so the cell cannot be misread as a pass. When neither
    half is present the metric was never measured, which is reported as an
    absence rather than as an unverifiable result.
    """
    if count is None and rate is None:
        return "—"
    if rate is None:
        return f"UNKNOWN（{unknown_reason}）"
    # ``bool`` is an int that never means a count, so it is rejected first.
    if isinstance(count, bool) or isinstance(rate, bool):
        return "無效資料"
    if not isinstance(count, (int, float)) or not isinstance(rate, (int, float)):
        return "無效資料"
    return f"{int(count)}（{float(rate):.2%}）"
