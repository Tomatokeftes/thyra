"""
Common test fixtures for msiconvert tests.
"""
import os
import pytest
import tempfile
import numpy as np
import shutil
from pathlib import Path
from lxml import etree
from pyimzml.ImzMLWriter import ImzMLWriter

# Root directory of the tests
TEST_DIR = Path(__file__).parent.resolve()
# Test data directory
DATA_DIR = TEST_DIR / "data"

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_reader():
    """Create a mock MSI reader for testing converters."""
    from msiconvert.core.base_reader import BaseMSIReader
    
    class MockMSIReader(BaseMSIReader):
        def __init__(self):
            self.closed = False
            
        def get_metadata(self):
            return {"source": "mock", "instrument": "test_instrument"}
            
        def get_dimensions(self):
            return (3, 3, 1)  # 3x3 grid, single z-plane
            
        def get_common_mass_axis(self):
            return np.linspace(100, 1000, 100)  # 100 mass values
            
        def iter_spectra(self):
            mass_axis = self.get_common_mass_axis()
            for x in range(3):
                for y in range(3):
                    # Create simple synthetic spectrum
                    intensities = np.zeros_like(mass_axis)
                    # Add a few peaks
                    intensities[x*10 + 20] = 100.0  # Peak varying by x position
                    intensities[y*10 + 50] = 200.0  # Peak varying by y position
                    yield ((x, y, 0), mass_axis, intensities)
                    
        def close(self):
            self.closed = True
    
    return MockMSIReader()

@pytest.fixture
def create_minimal_imzml(temp_dir):
    """
    Create a minimal imzML file for testing.
    Returns a tuple of (imzml_path, ibd_path, mzs, intensities)
    """
    imzml_path = temp_dir / "minimal.imzML"
    ibd_path = temp_dir / "minimal.ibd"
    
    # Create small sample data
    coordinates = [(1, 1, 1), (1, 2, 1), (2, 1, 1), (2, 2, 1)]  # 2x2 grid
    mzs = np.linspace(100, 1000, 50)  # 50 m/z values
    
    # Create different intensities for each pixel
    all_intensities = []
    for i, (x, y, z) in enumerate(coordinates):
        intensities = np.zeros_like(mzs)
        # Create a few peaks with position-dependent intensity
        intensities[10] = 100.0 * x  # Peak intensity depends on x
        intensities[30] = 150.0 * y  # Peak intensity depends on y
        all_intensities.append(intensities)
    
    # Write imzML file
    with ImzMLWriter(str(imzml_path), mode="processed") as writer:
        for i, (x, y, z) in enumerate(coordinates):
            writer.addSpectrum(mzs, all_intensities[i], (x, y, z))
    
    return imzml_path, ibd_path, mzs, all_intensities

@pytest.fixture
def mock_bruker_data(temp_dir):
    """
    Create a mock Bruker data directory structure.
    This is just for structure testing - won't contain real data.
    """
    bruker_dir = temp_dir / "mock.d"
    bruker_dir.mkdir(parents=True)
    
    # Create minimal analysis.tsf file
    with open(bruker_dir / "analysis.tsf", "w") as f:
        f.write("Mock Bruker TSF file for testing")
    
    # Create a fake SQLite DB file
    with open(bruker_dir / "analysis.sqlite", "w") as f:
        f.write("Mock SQLite DB file")
    
    return bruker_dir