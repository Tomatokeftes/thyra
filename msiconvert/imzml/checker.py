import logging
from pathlib import Path
from lxml.etree import iterparse
from .utils import get_imzml_pair

MZML_PREFIX = '{http://psi.hupo.org/ms/mzml}'
IMZML_UUID_ACCESSOR = 'IMS:1000080'


def check_uuid(imzml_path: Path, ibd_path: Path) -> bool:
    """Verify that a pair of imzML & ibd files have the same UUID.

    Args:
        imzml_path (Path): Path to the imzML file.
        ibd_path (Path): Path to the ibd file.

    Returns:
        bool: True if the UUIDs match, False otherwise.
    """
    try:        
        # Read the UUID from the ibd file as a lowercase hex string
        with open(ibd_path, mode='rb') as ibd:
            ibd_uuid = ibd.read(16).hex()

        # Parse the XML root element in the imzML file
        _, root = next(iterparse(str(imzml_path), events=['start']))

        # Locate the UUID accessor in the XML
        key = f'.//{MZML_PREFIX}cvParam[@accession="{IMZML_UUID_ACCESSOR}"]'
        element = root.find(key)
        if element is None:
            raise ValueError("Unable to find UUID in imzML file")

        # Extract and clean up UUID from the imzML file
        imzml_uuid = element.get('value')
        if imzml_uuid.startswith('{') and imzml_uuid.endswith('}'):
            imzml_uuid = imzml_uuid[1:-1]

        # Normalize UUID format by removing hyphens and converting to lowercase
        imzml_uuid = imzml_uuid.replace('-', '').lower()

        return imzml_uuid == ibd_uuid

    except StopIteration:
        # Empty XML file
        return False
    except OSError:
        # File not found or inaccessible
        return False
    except ValueError:
        # Parsing error
        return False
    except Exception as error:
        logging.error("Unexpected exception caught", exc_info=error)
        return False


class ImzMLChecker:
    """Standalone checker for validating imzML files."""

    @classmethod
    def match(cls, path: Path, imzml_filename: str = "sample.imzML", ibd_filename: str = "sample.ibd") -> bool:
        """Determine whether a given path points to a valid imzML and ibd file pair."""
        try:
            # Now passes default filenames
            pair = get_imzml_pair(path, imzml_filename, ibd_filename)
            if pair is None:
                return False
            # Check that the UUIDs match for the pair
            return check_uuid(*pair)
        except Exception:
            return False

