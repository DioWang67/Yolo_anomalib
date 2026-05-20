"""Security utilities for safe file operations.

This module provides path validation to prevent directory traversal attacks
and other security vulnerabilities related to file system access.
"""
from __future__ import annotations

import re
from pathlib import Path


class SecurityError(Exception):
    """Raised when a security check fails."""

    pass


class PathValidator:
    """Validate file paths to prevent directory traversal attacks.
    
    This validator ensures that all file paths accessed by the application
    are within allowed root directories, preventing attackers from accessing
    sensitive files outside the intended scope.
    
    Example:
        >>> validator = PathValidator(allowed_roots=[Path("/app/data")])
        >>> safe_path = validator.validate_path("/app/data/config.yaml")
        >>> # This would raise SecurityError:
        >>> # validator.validate_path("/etc/passwd")
    """

    def __init__(self, allowed_roots: list[Path]) -> None:
        """Initialize the path validator.
        
        Args:
            allowed_roots: List of root directories that are allowed to be accessed.
        """
        self.allowed_roots = [Path(root).resolve() for root in allowed_roots]

    def validate_path(
        self, path: str | Path, *, must_exist: bool = False
    ) -> Path:
        """Validate that a path is safe to access.

        Args:
            path: Path to validate (can be relative or absolute)
            must_exist: If True, verify that the path exists

        Returns:
            Resolved absolute path if validation passes

        Raises:
            SecurityError: If path is outside allowed root directories
            FileNotFoundError: If must_exist=True and path doesn't exist
            
        Example:
            >>> validator = PathValidator(allowed_roots=[Path("/app")])
            >>> # Safe path
            >>> validator.validate_path("/app/data/file.txt")
            PosixPath('/app/data/file.txt')
            >>> # Unsafe path (directory traversal attempt)
            >>> validator.validate_path("/app/../etc/passwd")
            Traceback (most recent call last):
            ...
            SecurityError: Access denied: /app/../etc/passwd is outside allowed directories
        """
        resolved = Path(path).resolve()

        # Check if path is within any of the allowed roots
        if not any(
            self._is_relative_to(resolved, root) for root in self.allowed_roots
        ):
            raise SecurityError(
                f"Access denied: {path} is outside allowed directories. "
                f"Allowed roots: {[str(r) for r in self.allowed_roots]}"
            )

        # Optionally verify the path exists
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        return resolved

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        """Check if path is relative to parent directory.
        
        This is a compatibility shim for Python < 3.9 which doesn't have
        Path.is_relative_to() built-in.
        
        Args:
            path: Path to check
            parent: Parent directory
            
        Returns:
            True if path is under parent, False otherwise
        """
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False


_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def safe_segment(value: object, *, field_name: str = "path segment") -> str:
    """Validate one filesystem path segment.

    Args:
        value: Segment value from product, area, status, detector, or similar input.
        field_name: Human-readable name used in error messages.

    Returns:
        The stripped segment string.

    Raises:
        SecurityError: If the value is empty or contains path separators,
            traversal markers, drive prefixes, or unsupported characters.
    """
    segment = str(value or "").strip()
    if not segment:
        raise SecurityError(f"Invalid {field_name}: empty value")
    if segment in {".", ".."}:
        raise SecurityError(f"Invalid {field_name}: traversal marker")
    if "/" in segment or "\\" in segment or ":" in segment:
        raise SecurityError(f"Invalid {field_name}: path separators are not allowed")
    if not _SAFE_SEGMENT_RE.fullmatch(segment):
        raise SecurityError(
            f"Invalid {field_name}: only letters, numbers, dot, underscore, "
            "and dash are allowed"
        )
    return segment


def ensure_subpath(
    path: str | Path, root: str | Path, *, must_exist: bool = False
) -> Path:
    """Resolve ``path`` and ensure it stays under ``root``.

    Args:
        path: Path to validate.
        root: Allowed root directory.
        must_exist: Whether the target must already exist.

    Returns:
        Resolved absolute path.

    Raises:
        SecurityError: If the path resolves outside ``root``.
        FileNotFoundError: If ``must_exist`` is true and the path does not exist.
    """
    validator = PathValidator([Path(root)])
    return validator.validate_path(path, must_exist=must_exist)


def resolve_output_dir(
    value: str | Path | None,
    *,
    base_dir: str | Path | None = None,
    allowed_root: str | Path | None = None,
    default_name: str = "Result",
) -> Path:
    """Resolve and validate an output directory.

    Relative output directories are resolved against ``base_dir`` when supplied,
    otherwise against ``allowed_root``. Absolute output directories are accepted
    only when they are under ``allowed_root``.

    Args:
        value: Raw configured output directory.
        base_dir: Base for relative paths.
        allowed_root: Directory that must contain the resolved output path.
        default_name: Fallback directory when ``value`` is empty.

    Returns:
        Resolved absolute output directory.

    Raises:
        SecurityError: If the resolved path escapes ``allowed_root``.
    """
    root = Path(allowed_root or PROJECT_ROOT).resolve()
    raw = str(value or "").strip() or default_name
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        base = Path(base_dir).resolve() if base_dir is not None else root
        candidate = base / candidate
    return ensure_subpath(candidate, root, must_exist=False)


# Global path validator instance
# This can be imported and used throughout the application
from core.path_utils import project_root
PROJECT_ROOT = project_root()

path_validator = PathValidator(
    allowed_roots=[
        PROJECT_ROOT,              # Config files at project root (config.yaml)
        PROJECT_ROOT / "models",   # Model weights directory
        PROJECT_ROOT / "Result",   # Output directory
        PROJECT_ROOT / "Runtime",  # Runtime directory
        PROJECT_ROOT / "MvImport", # Camera imports
        PROJECT_ROOT / "logs",     # Log files
    ]
)
