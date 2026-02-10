"""Model version parsing and validation utilities.

This module provides functions to extract version information from model filenames
and validate version compatibility.

Version Format: {model_name}_v{major}.{minor}.{patch}_{YYYYMMDD}.pt
Example: LED_best_v1.2.0_20260210.pt
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple


class ModelVersionError(Exception):
    """Raised when model version is incompatible or invalid."""
    pass


def parse_model_version(filename: str | Path) -> Tuple[int, int, int] | None:
    """Extract semantic version from model filename.
    
    Args:
        filename: Model filename or path (e.g., "LED_best_v1.2.0_20260210.pt")
    
    Returns:
        Tuple of (major, minor, patch) if version found, else None
    
    Examples:
        >>> parse_model_version("LED_best_v1.2.0_20260210.pt")
        (1, 2, 0)
        >>> parse_model_version("best.pt")  # Legacy format
        None
    """
    if isinstance(filename, Path):
        filename = filename.name
    
    # Pattern: _v{major}.{minor}.{patch}_
    pattern = r'_v(\d+)\.(\d+)\.(\d+)_'
    match = re.search(pattern, filename)
    
    if match:
        return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def version_to_string(version: Tuple[int, int, int]) -> str:
    """Convert version tuple to string format.
    
    Args:
        version: Tuple of (major, minor, patch)
    
    Returns:
        Version string (e.g., "1.2.0")
    
    Examples:
        >>> version_to_string((1, 2, 0))
        '1.2.0'
    """
    return f"{version[0]}.{version[1]}.{version[2]}"


def parse_version_string(version_str: str) -> Tuple[int, int, int]:
    """Parse version string to tuple.
    
    Args:
        version_str: Version string (e.g., "1.2.0")
    
    Returns:
        Tuple of (major, minor, patch)
    
    Raises:
        ValueError: If version string is invalid
    
    Examples:
        >>> parse_version_string("1.2.0")
        (1, 2, 0)
    """
    parts = version_str.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version string: {version_str}")
    
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as e:
        raise ValueError(f"Invalid version string: {version_str}") from e


def check_compatibility(
    current_version: Tuple[int, int, int],
    min_version: Tuple[int, int, int]
) -> bool:
    """Check if current version meets minimum requirement.
    
    Args:
        current_version: Current model version
        min_version: Minimum supported version
    
    Returns:
        True if compatible, False otherwise
    
    Examples:
        >>> check_compatibility((1, 2, 0), (1, 0, 0))
        True
        >>> check_compatibility((0, 9, 0), (1, 0, 0))
        False
    """
    return current_version >= min_version


def find_latest_version(weights_dir: Path, model_name: str) -> Path | None:
    """Find the latest versioned model in weights directory.
    
    Args:
        weights_dir: Directory containing model weights
        model_name: Model name prefix (e.g., "LED_best")
    
    Returns:
        Path to latest model file, or None if no versioned models found
    
    Examples:
        >>> weights_dir = Path("models/LED/A/yolo/weights")
        >>> find_latest_version(weights_dir, "LED_best")
        Path("models/LED/A/yolo/weights/LED_best_v1.2.0_20260210.pt")
    """
    if not weights_dir.exists():
        return None
    
    versioned_models = []
    for model_path in weights_dir.glob(f"{model_name}_v*.pt"):
        version = parse_model_version(model_path)
        if version:
            versioned_models.append((version, model_path))
    
    if not versioned_models:
        return None
    
    # Sort by version (descending)
    versioned_models.sort(key=lambda x: x[0], reverse=True)
    return versioned_models[0][1]


def generate_model_filename(
    model_name: str,
    version: Tuple[int, int, int] | str,
    date: str | None = None
) -> str:
    """Generate standardized model filename.
    
    Args:
        model_name: Base model name (e.g., "LED_best")
        version: Version tuple or string
        date: Date string (YYYYMMDD), defaults to today
    
    Returns:
        Standardized filename
    
    Examples:
        >>> generate_model_filename("LED_best", (1, 2, 0), "20260210")
        'LED_best_v1.2.0_20260210.pt'
        >>> generate_model_filename("Cable_detector", "1.0.0")
        'Cable_detector_v1.0.0_20260210.pt'  # Uses current date
    """
    if isinstance(version, str):
        version = parse_version_string(version)
    
    if date is None:
        from datetime import datetime
        date = datetime.now().strftime("%Y%m%d")
    
    version_str = version_to_string(version)
    return f"{model_name}_v{version_str}_{date}.pt"
