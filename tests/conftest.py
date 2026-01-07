# tests/conftest.py
import importlib
import sys
import tempfile
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

collect_ignore = []

try:
    importlib.import_module("jsonargparse")
except ModuleNotFoundError:
    collect_ignore.append("../core/performance_test.py")

@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch):
    """
    Automatically add the system temporary directory to the allowed roots
    of the global path_validator for all tests.
    This prevents 'SecurityError' when tests use 'tmp_path' fixture to create config files.
    """
    try:
        from core.security import path_validator
        
        # Get system temp dir (where pytest tmp_path lives)
        # On Windows: C:\Users\ADMIN~1\AppData\Local\Temp
        # On Linux: /tmp
        temp_dir = Path(tempfile.gettempdir()).resolve()
        
        # Add to allowed roots
        current_roots = list(path_validator.allowed_roots)
        current_roots.append(temp_dir)
        
        # Also add the specific pytest temp root if possible, 
        # but system temp should cover it as long as validator establishes parenthood correctly
        
        monkeypatch.setattr(path_validator, "allowed_roots", current_roots)
            
    except ImportError:
        pass
