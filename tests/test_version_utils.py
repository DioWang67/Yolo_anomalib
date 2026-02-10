"""Test suite for model version utilities."""

from pathlib import Path

import pytest

from core.version_utils import (
    ModelVersionError,
    check_compatibility,
    find_latest_version,
    generate_model_filename,
    parse_model_version,
    parse_version_string,
    version_to_string,
)


class TestParseModelVersion:
    """Test version parsing from filenames."""
    
    def test_parse_valid_version(self):
        """Should extract version from valid filename."""
        result = parse_model_version("LED_best_v1.2.0_20260210.pt")
        assert result == (1, 2, 0)
    
    def test_parse_path_object(self):
        """Should handle Path objects."""
        path = Path("models/LED/A/yolo/weights/LED_best_v1.2.0_20260210.pt")
        result = parse_model_version(path)
        assert result == (1, 2, 0)
    
    def test_parse_legacy_format(self):
        """Should return None for legacy non-versioned filenames."""
        result = parse_model_version("best.pt")
        assert result is None
    
    def test_parse_different_model_names(self):
        """Should work with various model name prefixes."""
        assert parse_model_version("Cable_detector_v2.0.0_20260101.pt") == (2, 0, 0)
        assert parse_model_version("PCBA_anomaly_v0.9.5_20250815.pt") == (0, 9, 5)
    
    def test_parse_high_version_numbers(self):
        """Should handle version numbers > 9."""
        result = parse_model_version("model_v10.25.100_20260101.pt")
        assert result == (10, 25, 100)


class TestVersionStringConversion:
    """Test version tuple <-> string conversion."""
    
    def test_version_to_string(self):
        """Should convert tuple to string."""
        assert version_to_string((1, 2, 0)) == "1.2.0"
        assert version_to_string((0, 0, 1)) == "0.0.1"
        assert version_to_string((10, 25, 100)) == "10.25.100"
    
    def test_parse_version_string(self):
        """Should parse valid version strings."""
        assert parse_version_string("1.2.0") == (1, 2, 0)
        assert parse_version_string("0.0.1") == (0, 0, 1)
    
    def test_parse_invalid_version_string(self):
        """Should raise ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Invalid version string"):
            parse_version_string("1.2")
        with pytest.raises(ValueError, match="Invalid version string"):
            parse_version_string("1.2.0.0")
        with pytest.raises(ValueError, match="Invalid version string"):
            parse_version_string("a.b.c")


class TestCompatibilityCheck:
    """Test version compatibility validation."""
    
    def test_compatible_versions(self):
        """Should return True for compatible versions."""
        assert check_compatibility((1, 2, 0), (1, 0, 0)) is True
        assert check_compatibility((2, 0, 0), (1, 5, 0)) is True
        assert check_compatibility((1, 0, 0), (1, 0, 0)) is True  # Equal
    
    def test_incompatible_versions(self):
        """Should return False for incompatible versions."""
        assert check_compatibility((0, 9, 0), (1, 0, 0)) is False
        assert check_compatibility((1, 0, 0), (1, 1, 0)) is False
        assert check_compatibility((1, 2, 3), (1, 2, 4)) is False


class TestFindLatestVersion:
    """Test finding latest versioned model in directory."""
    
    def test_find_latest_among_multiple(self, tmp_path):
        """Should return the latest version."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        
        # Create multiple versions
        (weights_dir / "LED_best_v1.0.0_20260101.pt").touch()
        (weights_dir / "LED_best_v1.2.0_20260201.pt").touch()
        latest = weights_dir / "LED_best_v1.1.0_20260115.pt"
        latest.touch()  # Not latest by date, but v1.2.0 wins
        
        result = find_latest_version(weights_dir, "LED_best")
        assert result.name == "LED_best_v1.2.0_20260201.pt"
    
    def test_no_versioned_models(self, tmp_path):
        """Should return None if no versioned models exist."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir()
        (weights_dir / "best.pt").touch()  # Legacy format
        
        result = find_latest_version(weights_dir, "LED_best")
        assert result is None
    
    def test_directory_not_exists(self, tmp_path):
        """Should return None if directory doesn't exist."""
        result = find_latest_version(tmp_path / "nonexistent", "model")
        assert result is None


class TestGenerateFilename:
    """Test model filename generation."""
    
    def test_generate_with_tuple_version(self):
        """Should generate filename from version tuple."""
        result = generate_model_filename("LED_best", (1, 2, 0), "20260210")
        assert result == "LED_best_v1.2.0_20260210.pt"
    
    def test_generate_with_string_version(self):
        """Should accept version as string."""
        result = generate_model_filename("Cable_detector", "2.0.1", "20260101")
        assert result == "Cable_detector_v2.0.1_20260101.pt"
    
    def test_generate_without_date(self):
        """Should use current date if not provided."""
        from datetime import datetime
        result = generate_model_filename("model", (1, 0, 0))
        expected_date = datetime.now().strftime("%Y%m%d")
        assert expected_date in result
        assert result.startswith("model_v1.0.0_")
        assert result.endswith(".pt")
