from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QTimer


def load_image_with_retry(
    viewer,
    path: Optional[str],
    *,
    attempts: int = 5,
    delay_ms: int = 200,
    on_fail: Optional[Callable[[], None]] = None,
) -> None:
    """Attempt to display an image with retries to handle async file writes."""

    if not path:
        if on_fail:
            on_fail()
        else:
            viewer.setText(f"無法顯示{getattr(viewer, 'title', '')}")
        return

    def _attempt(remaining: int) -> None:
        from pathlib import Path

        candidate = Path(path)
        if candidate.exists():
            viewer.set_image(str(candidate))
            return
        if remaining <= 0:
            if on_fail:
                on_fail()
            else:
                viewer.setText(f"無法載入{getattr(viewer, 'title', '')}")
            return
        QTimer.singleShot(delay_ms, lambda: _attempt(remaining - 1))

    _attempt(max(0, int(attempts)))
