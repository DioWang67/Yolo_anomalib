"""Security feature tests for path validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.security import PathValidator, SecurityError


class TestPathValidator:
    """Test suite for PathValidator security features."""

    def test_allows_valid_path_within_allowed_root(self, tmp_path):
        """Test that valid paths within allowed roots are accepted."""
        validator = PathValidator(allowed_roots=[tmp_path])

        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Should not raise
        result = validator.validate_path(test_file, must_exist=True)
        assert result == test_file.resolve()

    def test_allows_subdirectory_path(self, tmp_path):
        """Test that paths in subdirectories are allowed."""
        validator = PathValidator(allowed_roots=[tmp_path])

        subdir = tmp_path / "subdir" / "nested"
        subdir.mkdir(parents=True)
        test_file = subdir / "file.txt"
        test_file.write_text("nested content")

        result = validator.validate_path(test_file, must_exist=True)
        assert result == test_file.resolve()

    def test_blocks_directory_traversal_with_dotdot(self, tmp_path):
        """Test that directory traversal using .. is blocked."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        validator = PathValidator(allowed_roots=[safe_dir])

        # Create a file outside the allowed root
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret")

        # Try to access it using directory traversal
        malicious_path = safe_dir / ".." / "secret.txt"

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_path(malicious_path)

        assert "outside allowed directories" in str(exc_info.value)

    def test_blocks_absolute_path_outside_allowed_roots(self, tmp_path):
        """Test that absolute paths outside allowed roots are blocked."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        validator = PathValidator(allowed_roots=[allowed])

        forbidden = tmp_path / "forbidden"
        forbidden.mkdir()
        outside_file = forbidden / "file.txt"
        outside_file.write_text("forbidden content")

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_path(outside_file)

        assert "outside allowed directories" in str(exc_info.value)

    def test_must_exist_flag_enforces_existence(self, tmp_path):
        """Test that must_exist=True raises error for non-existent files."""
        validator = PathValidator(allowed_roots=[tmp_path])

        non_existent = tmp_path / "does_not_exist.txt"

        # Should raise FileNotFoundError with must_exist=True
        with pytest.raises(FileNotFoundError):
            validator.validate_path(non_existent, must_exist=True)

        # Should not raise without must_exist
        result = validator.validate_path(non_existent, must_exist=False)
        assert result == non_existent.resolve()

    def test_multiple_allowed_roots(self, tmp_path):
        """Test validator with multiple allowed root directories."""
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        root1.mkdir()
        root2.mkdir()

        validator = PathValidator(allowed_roots=[root1, root2])

        # Files in both roots should be allowed
        file1 = root1 / "file1.txt"
        file1.write_text("content1")
        result1 = validator.validate_path(file1, must_exist=True)
        assert result1 == file1.resolve()

        file2 = root2 / "file2.txt"
        file2.write_text("content2")
        result2 = validator.validate_path(file2, must_exist=True)
        assert result2 == file2.resolve()

        # File outside both roots should be blocked
        outside = tmp_path / "outside.txt"
        outside.write_text("outside")
        with pytest.raises(SecurityError):
            validator.validate_path(outside)

    def test_handles_relative_paths(self, tmp_path, monkeypatch):
        """Test that relative paths are resolved correctly."""
        validator = PathValidator(allowed_roots=[tmp_path])

        # Change to the tmp_path directory
        monkeypatch.chdir(tmp_path)

        test_file = tmp_path / "relative.txt"
        test_file.write_text("relative content")

        # Use relative path
        result = validator.validate_path("relative.txt", must_exist=True)
        assert result == test_file.resolve()

    def test_blocks_symlink_escape(self, tmp_path):
        """Test that symlinks cannot be used to escape allowed roots."""
        safe_dir = tmp_path / "safe"
        safe_dir.mkdir()
        validator = PathValidator(allowed_roots=[safe_dir])

        # Create a file outside the safe directory
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret data")

        # Try to create a symlink inside safe_dir pointing outside
        symlink_path = safe_dir / "link_to_secret"

        try:
            symlink_path.symlink_to(outside_file)
        except OSError:
            # Skip test if symlinks are not supported (e.g., Windows without admin)
            pytest.skip("Symlinks not supported on this system")

        # The symlink itself is in safe_dir, but it resolves to outside
        # The validator should block access based on the resolved path
        with pytest.raises(SecurityError):
            validator.validate_path(symlink_path)

    def test_error_message_includes_allowed_roots(self, tmp_path):
        """Test that error messages include the list of allowed roots."""
        root1 = tmp_path / "root1"
        root2 = tmp_path / "root2"
        root1.mkdir()
        root2.mkdir()

        validator = PathValidator(allowed_roots=[root1, root2])

        outside = tmp_path / "outside.txt"
        outside.write_text("outside")

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_path(outside)

        error_msg = str(exc_info.value)
        assert "Allowed roots:" in error_msg
        # Normalize paths for comparison (handle both Windows and Unix separators)
        root1_str = str(root1.resolve()).replace("\\", "\\\\")
        root2_str = str(root2.resolve()).replace("\\", "\\\\")
        assert root1_str in error_msg or str(root1.resolve()) in error_msg
        assert root2_str in error_msg or str(root2.resolve()) in error_msg


class TestGlobalPathValidator:
    """Test the global path_validator instance."""

    def test_global_validator_allows_project_paths(self):
        """Test that the global validator allows project paths."""
        from core.security import PROJECT_ROOT, path_validator

        # Should allow paths in project root
        test_path = PROJECT_ROOT / "test_file.txt"
        result = path_validator.validate_path(test_path, must_exist=False)
        assert result == test_path.resolve()

    def test_global_validator_allows_models_directory(self):
        """Test that the global validator allows models directory."""
        from core.security import PROJECT_ROOT, path_validator

        models_path = PROJECT_ROOT / "models" / "test" / "model.pt"
        result = path_validator.validate_path(models_path, must_exist=False)
        assert result == models_path.resolve()

    def test_global_validator_allows_result_directory(self):
        """Test that the global validator allows Result directory."""
        from core.security import PROJECT_ROOT, path_validator

        result_path = PROJECT_ROOT / "Result" / "output.jpg"
        result = path_validator.validate_path(result_path, must_exist=False)
        assert result == result_path.resolve()

    def test_global_validator_blocks_system_paths(self):
        """Test that the global validator blocks system paths."""
        import platform

        from core.security import path_validator

        # Try to access a system file
        if platform.system() == "Windows":
            system_path = Path("C:/Windows/System32/config/SAM")
        else:
            system_path = Path("/etc/passwd")

        with pytest.raises(SecurityError):
            path_validator.validate_path(system_path)
