# tests/unit/readers/test_bruker_folder_structure.py
"""Tests for BrukerFolderStructure abstraction."""

import pytest

from thyra.readers.bruker.folder_structure import (
    BrukerFolderInfo,
    BrukerFolderStructure,
    BrukerFormat,
)


class TestBrukerFormat:
    """Tests for BrukerFormat enum."""

    def test_format_values(self):
        """Test that format enum has expected values."""
        assert BrukerFormat.TIMSTOF.value == "timstof"
        assert BrukerFormat.FLEXIMAGING.value == "fleximaging"
        assert BrukerFormat.UNKNOWN.value == "unknown"


class TestBrukerFolderStructure:
    """Tests for BrukerFolderStructure class."""

    def test_detect_timstof_tsf(self, tmp_path):
        """Test detection of timsTOF TSF format."""
        # Create .d directory with analysis.tsf
        d_dir = tmp_path / "test.d"
        d_dir.mkdir()
        (d_dir / "analysis.tsf").touch()
        (d_dir / "analysis.tsf_bin").touch()

        folder = BrukerFolderStructure(d_dir)
        info = folder.analyze()

        assert info.format == BrukerFormat.TIMSTOF
        assert info.data_path == d_dir

    def test_detect_timstof_tdf(self, tmp_path):
        """Test detection of timsTOF TDF format."""
        # Create .d directory with analysis.tdf
        d_dir = tmp_path / "test.d"
        d_dir.mkdir()
        (d_dir / "analysis.tdf").touch()
        (d_dir / "analysis.tdf_bin").touch()

        folder = BrukerFolderStructure(d_dir)
        info = folder.analyze()

        assert info.format == BrukerFormat.TIMSTOF
        assert info.data_path == d_dir

    def test_detect_fleximaging(self, tmp_path):
        """Test detection of FlexImaging format."""
        # Create FlexImaging folder structure
        data_dir = tmp_path / "fleximaging_data"
        data_dir.mkdir()
        (data_dir / "sample.dat").touch()
        (data_dir / "sample_poslog.txt").touch()
        (data_dir / "sample_info.txt").touch()

        folder = BrukerFolderStructure(data_dir)
        info = folder.analyze()

        assert info.format == BrukerFormat.FLEXIMAGING
        assert info.data_path == data_dir

    def test_detect_unknown_format(self, tmp_path):
        """Test detection returns UNKNOWN for unrecognized format."""
        # Create empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        folder = BrukerFolderStructure(empty_dir)
        info = folder.analyze()

        assert info.format == BrukerFormat.UNKNOWN

    def test_detect_timstof_from_parent(self, tmp_path):
        """Test detection of timsTOF from parent folder."""
        # Create parent folder containing .d subfolder
        parent = tmp_path / "experiment"
        parent.mkdir()
        d_dir = parent / "data.d"
        d_dir.mkdir()
        (d_dir / "analysis.tsf").touch()

        folder = BrukerFolderStructure(parent)
        info = folder.analyze()

        assert info.format == BrukerFormat.TIMSTOF
        assert info.data_path == d_dir

    def test_detect_fleximaging_from_parent(self, tmp_path):
        """Test detection of FlexImaging from parent folder."""
        # Create parent folder containing FlexImaging subfolder
        parent = tmp_path / "experiment"
        parent.mkdir()
        data_dir = parent / "data"
        data_dir.mkdir()
        (data_dir / "sample.dat").touch()
        (data_dir / "sample_poslog.txt").touch()
        (data_dir / "sample_info.txt").touch()

        folder = BrukerFolderStructure(parent)
        info = folder.analyze()

        assert info.format == BrukerFormat.FLEXIMAGING
        assert info.data_path == data_dir

    def test_find_optical_images(self, tmp_path):
        """Test finding optical TIFF images."""
        # Create FlexImaging structure with TIFF files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "sample.dat").touch()
        (data_dir / "sample_poslog.txt").touch()
        (data_dir / "sample_info.txt").touch()
        (data_dir / "optical_0000.tif").touch()
        (data_dir / "optical_0001.tiff").touch()

        folder = BrukerFolderStructure(data_dir)
        info = folder.analyze()

        assert len(info.optical_images) == 2
        assert any("optical_0000.tif" in str(p) for p in info.optical_images)
        assert any("optical_0001.tiff" in str(p) for p in info.optical_images)

    def test_find_optical_images_at_parent(self, tmp_path):
        """Test finding optical images in parent folder."""
        # Create structure where TIFFs are at parent level
        parent = tmp_path / "experiment"
        parent.mkdir()
        (parent / "optical.tif").touch()

        d_dir = parent / "data.d"
        d_dir.mkdir()
        (d_dir / "analysis.tsf").touch()

        folder = BrukerFolderStructure(parent)
        info = folder.analyze()

        assert len(info.optical_images) == 1
        assert "optical.tif" in str(info.optical_images[0])

    def test_find_teaching_points_file(self, tmp_path):
        """Test finding .mis teaching points file."""
        # Create FlexImaging structure with .mis file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "sample.dat").touch()
        (data_dir / "sample_poslog.txt").touch()
        (data_dir / "sample_info.txt").touch()
        (data_dir / "sample.mis").touch()

        folder = BrukerFolderStructure(data_dir)
        info = folder.analyze()

        assert info.teaching_points_file is not None
        assert info.teaching_points_file.name == "sample.mis"

    def test_find_metadata_files_fleximaging(self, tmp_path):
        """Test finding FlexImaging metadata files."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "sample.dat").touch()
        (data_dir / "sample_poslog.txt").touch()
        (data_dir / "sample_info.txt").touch()
        (data_dir / "sample.mis").touch()

        folder = BrukerFolderStructure(data_dir)
        info = folder.analyze()

        assert "data" in info.metadata_files
        assert "poslog" in info.metadata_files
        assert "info" in info.metadata_files
        assert "mis" in info.metadata_files

    def test_find_metadata_files_timstof(self, tmp_path):
        """Test finding timsTOF metadata files."""
        d_dir = tmp_path / "test.d"
        d_dir.mkdir()
        (d_dir / "analysis.tsf").touch()
        (d_dir / "analysis.tsf_bin").touch()

        folder = BrukerFolderStructure(d_dir)
        info = folder.analyze()

        assert "tsf" in info.metadata_files
        assert "tsf_bin" in info.metadata_files

    def test_nonexistent_path_raises(self, tmp_path):
        """Test that nonexistent path raises ValueError."""
        nonexistent = tmp_path / "nonexistent"

        folder = BrukerFolderStructure(nonexistent)
        with pytest.raises(ValueError, match="does not exist"):
            folder.analyze()

    def test_classmethod_detect_format(self, tmp_path):
        """Test classmethod detect_format."""
        d_dir = tmp_path / "test.d"
        d_dir.mkdir()
        (d_dir / "analysis.tdf").touch()

        fmt = BrukerFolderStructure.detect_format(d_dir)
        assert fmt == BrukerFormat.TIMSTOF

    def test_classmethod_is_bruker_data(self, tmp_path):
        """Test classmethod is_bruker_data."""
        # Valid Bruker data
        d_dir = tmp_path / "test.d"
        d_dir.mkdir()
        (d_dir / "analysis.tsf").touch()

        assert BrukerFolderStructure.is_bruker_data(d_dir) is True

        # Invalid data
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        assert BrukerFolderStructure.is_bruker_data(empty_dir) is False

    def test_analyze_caching(self, tmp_path):
        """Test that analyze() result is cached."""
        d_dir = tmp_path / "test.d"
        d_dir.mkdir()
        (d_dir / "analysis.tsf").touch()

        folder = BrukerFolderStructure(d_dir)
        info1 = folder.analyze()
        info2 = folder.analyze()

        # Should be the same object (cached)
        assert info1 is info2


class TestBrukerFolderInfo:
    """Tests for BrukerFolderInfo dataclass."""

    def test_dataclass_fields(self, tmp_path):
        """Test BrukerFolderInfo has expected fields."""
        info = BrukerFolderInfo(
            path=tmp_path,
            format=BrukerFormat.TIMSTOF,
            data_path=tmp_path / "data.d",
            optical_images=[tmp_path / "image.tif"],
            teaching_points_file=tmp_path / "points.mis",
            metadata_files={"tdf": tmp_path / "analysis.tdf"},
        )

        assert info.path == tmp_path
        assert info.format == BrukerFormat.TIMSTOF
        assert info.data_path == tmp_path / "data.d"
        assert len(info.optical_images) == 1
        assert info.teaching_points_file == tmp_path / "points.mis"
        assert "tdf" in info.metadata_files

    def test_default_values(self, tmp_path):
        """Test BrukerFolderInfo default values."""
        info = BrukerFolderInfo(
            path=tmp_path,
            format=BrukerFormat.UNKNOWN,
            data_path=tmp_path,
        )

        assert info.optical_images == []
        assert info.teaching_points_file is None
        assert info.metadata_files == {}
