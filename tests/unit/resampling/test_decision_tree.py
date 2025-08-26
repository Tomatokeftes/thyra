"""
Tests for simplified DecisionTree with only timsTOF support - Phase 3.
"""

import pytest

from thyra.resampling.decision_tree import ResamplingDecisionTree
from thyra.resampling.types import ResamplingMethod


class TestSimplifiedDecisionTree:
    """Test simplified decision tree with only timsTOF support."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tree = ResamplingDecisionTree()

    def test_no_metadata_raises_error(self):
        """Test that no metadata raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            self.tree.select_strategy(None)

        assert "metadata not yet implemented" in str(exc_info.value)
        assert "timsTOF detection is supported" in str(exc_info.value)

        with pytest.raises(NotImplementedError):
            self.tree.select_strategy({})

    def test_timstof_detection_only_bruker_metadata(self):
        """Test that timsTOF detection only works via Bruker metadata."""
        # These should NOT be detected as timsTOF (only Bruker metadata detection)
        timstof_names = [
            "timsTOF Pro 2",
            "Bruker timsTOF",
            "TIMS-TOF CCS",
            "tims tof flex",
            "TimsTOF Impact II",
        ]

        for name in timstof_names:
            metadata = {"instrument_name": name}
            with pytest.raises(NotImplementedError) as exc_info:
                self.tree.select_strategy(metadata)
            assert name in str(exc_info.value)
            assert "not yet implemented" in str(exc_info.value)

    def test_imzml_centroid_spectrum_detection(self):
        """Test exact ImzML centroid spectrum detection."""
        # Test exact spectrum_type match
        metadata = {"spectrum_type": "centroid spectrum"}
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

        # Test cvParam match
        metadata = {
            "cvParams": [
                {
                    "accession": "MS:1000127",
                    "cvRef": "MS",
                    "name": "centroid spectrum",
                }
            ]
        }
        method = self.tree.select_strategy(metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

    def test_non_exact_spectrum_types_raise_error(self):
        """Test that non-exact spectrum types raise NotImplementedError."""
        non_exact_types = [
            "profile spectrum",
            "profile",
            "continuum",
            "raw",
            "continuous",
            "unprocessed",
            "centroided",
            "peak picked",
            "peaks",
        ]

        for spec_type in non_exact_types:
            metadata = {"spectrum_type": spec_type}
            with pytest.raises(NotImplementedError) as exc_info:
                self.tree.select_strategy(metadata)

            assert "not yet implemented" in str(exc_info.value)

    def test_non_timstof_instruments_raise_error(self):
        """Test that non-timsTOF instruments raise NotImplementedError."""
        non_timstof_instruments = [
            "Orbitrap Fusion Lumos",
            "Q Exactive HF-X",
            "LTQ Orbitrap Velos",
            "Waters Synapt G2-Si",
            "Bruker 12T FT-ICR",
            "Unknown MS Instrument",
        ]

        for instrument in non_timstof_instruments:
            metadata = {"instrument_name": instrument}
            with pytest.raises(NotImplementedError) as exc_info:
                self.tree.select_strategy(metadata)

            assert instrument in str(exc_info.value)
            assert "not yet implemented" in str(exc_info.value)
            assert "timsTOF detection is supported" in str(exc_info.value)

    def test_bruker_metadata_timstof_detection(self):
        """Test timsTOF detection from Bruker-specific metadata."""
        # Test with exact instrument name "timsTOF Maldi 2"
        bruker_timstof_metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF Maldi 2"}
        }

        method = self.tree.select_strategy(bruker_timstof_metadata)
        assert method == ResamplingMethod.NEAREST_NEIGHBOR

        # Test with other timsTOF variants should NOT be detected via Bruker metadata
        other_timstof_metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF Pro 2"}  # Different variant
        }
        with pytest.raises(NotImplementedError):
            self.tree.select_strategy(other_timstof_metadata)

    def test_non_bruker_metadata_raises_error(self):
        """Test that non-Bruker metadata without timsTOF raises error."""
        non_bruker_metadata = {
            "instrument_name": "Q Exactive",
            "file_format": "mzML",
            "acquisition_mode": "profile",
        }

        with pytest.raises(NotImplementedError):
            self.tree.select_strategy(non_bruker_metadata)

    def test_metadata_key_variants_raise_error(self):
        """Test different metadata key variants for instrument name all raise error."""
        instrument_key_variants = [
            "instrument_name",
            "instrument",
            "instrument_model",
            "InstrumentName",
            "Instrument",
            "InstrumentModel",
            "ms_instrument_name",
            "device_name",
        ]

        for key in instrument_key_variants:
            metadata = {key: "timsTOF Pro"}
            with pytest.raises(NotImplementedError):
                self.tree.select_strategy(metadata)

    def test_empty_instrument_name_raises_error(self):
        """Test that empty instrument names raise NotImplementedError."""
        empty_names = ["", "   ", None]

        for name in empty_names:
            metadata = {"instrument_name": name}
            with pytest.raises(NotImplementedError):
                self.tree.select_strategy(metadata)


class TestDecisionTreeHelperMethods:
    """Test helper methods of simplified DecisionTree."""

    def setup_method(self):
        """Setup test fixtures."""
        self.tree = ResamplingDecisionTree()

    def test_extract_instrument_name(self):
        """Test instrument name extraction."""
        metadata = {"instrument_name": "  timsTOF Pro 2  "}
        name = self.tree._extract_instrument_name(metadata)
        assert name == "timsTOF Pro 2"  # Should be stripped

        # Test None values
        metadata = {"instrument_name": None}
        name = self.tree._extract_instrument_name(metadata)
        assert name is None

        # Test empty metadata
        name = self.tree._extract_instrument_name({})
        assert name is None

    def test_extract_spectrum_type(self):
        """Test spectrum type extraction."""
        metadata = {"spectrum_type": "  centroid spectrum  "}
        spec_type = self.tree._extract_spectrum_type(metadata)
        assert spec_type == "centroid spectrum"  # Should be stripped and lowercased

        # Test None values
        metadata = {"spectrum_type": None}
        spec_type = self.tree._extract_spectrum_type(metadata)
        assert spec_type is None

        # Test empty metadata
        spec_type = self.tree._extract_spectrum_type({})
        assert spec_type is None

    def test_is_imzml_centroid_spectrum(self):
        """Test exact ImzML centroid spectrum detection."""
        # Test exact spectrum_type match
        metadata = {"spectrum_type": "centroid spectrum"}
        assert self.tree._is_imzml_centroid_spectrum(metadata)

        # Test cvParam match
        metadata = {
            "cvParams": [
                {
                    "accession": "MS:1000127",
                    "cvRef": "MS",
                    "name": "centroid spectrum",
                }
            ]
        }
        assert self.tree._is_imzml_centroid_spectrum(metadata)

        # Test non-exact matches should return False
        non_exact_metadata = [
            {"spectrum_type": "centroided"},
            {"spectrum_type": "profile spectrum"},
            {"cvParams": [{"name": "profile spectrum"}]},
            {},
        ]

        for metadata in non_exact_metadata:
            assert not self.tree._is_imzml_centroid_spectrum(metadata)

    def test_is_bruker_metadata(self):
        """Test Bruker metadata detection."""
        bruker_metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF"},
            "AcquisitionKeys": {},
        }
        assert self.tree._is_bruker_metadata(bruker_metadata)

        non_bruker_metadata = {
            "instrument_name": "Orbitrap",
            "file_format": "mzML",
        }
        assert not self.tree._is_bruker_metadata(non_bruker_metadata)

    def test_detect_timstof_from_bruker_metadata(self):
        """Test timsTOF detection from Bruker metadata."""
        # Test with exact instrument name "timsTOF Maldi 2"
        timstof_maldi_metadata = {
            "GlobalMetadata": {"InstrumentName": "timsTOF Maldi 2"}
        }
        assert self.tree._detect_timstof_from_bruker_metadata(timstof_maldi_metadata)

        # Test with other instrument names should NOT be detected
        other_metadata = {"GlobalMetadata": {"InstrumentName": "timsTOF Pro 2"}}
        assert not self.tree._detect_timstof_from_bruker_metadata(other_metadata)

        non_timstof_metadata = {
            "GlobalMetadata": {"InstrumentName": "Quadrupole LC-MS"}
        }
        assert not self.tree._detect_timstof_from_bruker_metadata(non_timstof_metadata)

        # Test without InstrumentName should not detect
        empty_metadata = {"GlobalMetadata": {"SomeOtherKey": "value"}}
        assert not self.tree._detect_timstof_from_bruker_metadata(empty_metadata)
