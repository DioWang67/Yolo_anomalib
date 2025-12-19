import os
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

collect_ignore = []

try:
    importlib.import_module("jsonargparse")
except ModuleNotFoundError:
    collect_ignore.append("../core/performance_test.py")

# Auto-detect QT_API for Local Windows environments where discovery fails
if os.environ.get("QT_API") is None and os.environ.get("PYTEST_QT_API") is None:
    try:
        importlib.import_module("PyQt5")
        os.environ["QT_API"] = "pyqt5"
    except ImportError:
        pass
