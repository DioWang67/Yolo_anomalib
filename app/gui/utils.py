from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QTimer


def _image_error(viewer, kind: str) -> str:
    """Return a localized image-load fallback message."""
    language = getattr(viewer, "_language", "en")
    title = getattr(viewer, "_title", "")
    if language == "zh":
        prefix = "無法顯示" if kind == "display" else "無法載入"
        return f"{prefix}{title}"
    prefix = "Unable to display" if kind == "display" else "Unable to load"
    return f"{prefix} {title}".strip()


def load_image_with_retry(
    viewer,
    path: str | None,
    *,
    attempts: int = 2,
    delay_ms: int = 150,
    on_fail: Callable[[], None] | None = None,
) -> None:
    """Attempt to display an image with retries to handle async file writes."""
    if not path:
        if on_fail:
            on_fail()
        else:
            viewer.setText(_image_error(viewer, "display"))
        return

    token: int = getattr(viewer, "_load_token", 0)

    def _attempt(remaining: int) -> None:
        from pathlib import Path

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
                viewer.setText(_image_error(viewer, "load"))
            return
        QTimer.singleShot(delay_ms, lambda: _attempt(remaining - 1))

    _attempt(max(0, int(attempts)))
