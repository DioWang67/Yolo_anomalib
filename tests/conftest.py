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
