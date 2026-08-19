"""Entrypoint for the standalone model-acceptance application."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from app.acceptance.window import ModelAcceptanceWindow
from core.path_utils import project_root


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("YOLO Model Acceptance")
    window = ModelAcceptanceWindow(project_root=Path(project_root()))
    window.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
