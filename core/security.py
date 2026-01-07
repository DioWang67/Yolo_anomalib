"""Security utilities for safe file operations.

This module provides path validation to prevent directory traversal attacks
and other security vulnerabilities related to file system access.
"""
from __future__ import annotations

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


# Global path validator instance
# This can be imported and used throughout the application
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

path_validator = PathValidator(
    allowed_roots=[
        PROJECT_ROOT,  # Entire project root
        PROJECT_ROOT / "models",  # Model weights directory
        PROJECT_ROOT / "Result",  # Output directory
        PROJECT_ROOT / "Runtime",  # Runtime directory
        PROJECT_ROOT / "MvImport",  # Camera imports
    ]
)
