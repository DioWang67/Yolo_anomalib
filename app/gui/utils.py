from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QTimer


def load_image_with_retry(
    viewer,
    path: str | None,
    *,
    attempts: int = 2,
    delay_ms: int = 150,
    on_fail: Callable[[], None] | None = None,
) -> None:
    """Attempt to display an image with retries to handle async file writes.

    Uses the viewer's ``_load_token`` to cancel retries that are superseded
    by a newer load request or a ``clear()`` call.  This prevents stale
    QTimer callbacks from overwriting images loaded by a subsequent run.
    """
    if not path:
        if on_fail:
            on_fail()
        else:
            viewer.setText(f"無法顯示{getattr(viewer, '_title', '')}")
        return

    # Snapshot the token at the moment this load was requested.
    # ImageViewer.set_image() will increment the token if called later,
    # making this snapshot stale and causing _attempt() to abort.
    token: int = getattr(viewer, "_load_token", 0)

    def _attempt(remaining: int) -> None:
        from pathlib import Path

        # Abort if the viewer has been cleared or a newer load started
        if getattr(viewer, "_load_token", 0) != token:
            return

        candidate = Path(path)
        if candidate.exists():
            viewer.set_image(str(candidate))
            return
        if remaining <= 0:
            if getattr(viewer, "_load_token", 0) != token:
                return
            if on_fail:
                on_fail()
            else:
                viewer.setText(f"無法載入{getattr(viewer, '_title', '')}")
            return
        QTimer.singleShot(delay_ms, lambda: _attempt(remaining - 1))

    _attempt(max(0, int(attempts)))
