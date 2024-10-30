import pytest
from pathlib import Path
from lxml import etree
from msiconvert.imzml.checker import ImzMLChecker, check_uuid

MZML_PREFIX = '{http://psi.hupo.org/ms/mzml}'
IMZML_UUID_ACCESSOR = 'IMS:1000080'

@pytest.fixture
def sample_files(tmp_path):
    """Creates a temporary imzML and ibd file pair with matching UUIDs."""
    imzml_path = tmp_path / "sample.imzml"
    ibd_path = tmp_path / "sample.ibd"
    
    # Generate UUID for the files
    uuid = "12345678-1234-5678-1234-567812345678"

    # Write the ibd file with the UUID in binary format
    with open(ibd_path, "wb") as ibd_file:
        ibd_file.write(bytes.fromhex(uuid.replace("-", "")))

    # Write a minimal imzML XML structure containing the UUID
    root = etree.Element(f"{MZML_PREFIX}mzML", nsmap={None: "http://psi.hupo.org/ms/mzml"})
    cvParam = etree.SubElement(root, f"{MZML_PREFIX}cvParam", accession=IMZML_UUID_ACCESSOR, value=f"{{{uuid}}}")
    tree = etree.ElementTree(root)
    with open(imzml_path, "wb") as imzml_file:
        tree.write(imzml_file, xml_declaration=True, encoding="UTF-8", pretty_print=True)

    return imzml_path, ibd_path

def test_check_uuid_matches(sample_files):
    """Test that check_uuid returns True when the UUIDs match."""
    imzml_path, ibd_path = sample_files
    assert check_uuid(imzml_path, ibd_path) == True

def test_check_uuid_no_match(sample_files):
    """Test that check_uuid returns False when the UUIDs do not match."""
    imzml_path, ibd_path = sample_files

    # Modify the ibd file with a different UUID
    different_uuid = "87654321-4321-8765-4321-876543218765"
    with open(ibd_path, "wb") as ibd_file:
        ibd_file.write(bytes.fromhex(different_uuid.replace("-", "")))

    assert check_uuid(imzml_path, ibd_path) == False

def test_check_uuid_missing_ibd_file(sample_files):
    """Test that check_uuid returns False when the ibd file is missing."""
    imzml_path, ibd_path = sample_files

    # Remove the ibd file to simulate a missing file scenario
    ibd_path.unlink()
    assert check_uuid(imzml_path, ibd_path) == False

def test_imzml_checker_match(sample_files):
    """Test that ImzMLChecker.match returns True for a valid imzML-ibd pair."""
    imzml_path, _ = sample_files
    assert ImzMLChecker.match(imzml_path.parent) == True

def test_imzml_checker_no_match(sample_files):
    """Test that ImzMLChecker.match returns False if no valid pair is found."""
    imzml_path, ibd_path = sample_files

    # Remove the ibd file to break the pair
    ibd_path.unlink()

    assert ImzMLChecker.match(imzml_path.parent) == False
