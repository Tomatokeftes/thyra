import abc
import zarr
from typing import List

class BaseMSIConverter(abc.ABC):
    """Abstract base class for MSI data converters."""

    def __init__(self, root: zarr.Group, name: str, **kwargs):
        self.root = root
        self.name = name

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        """Return the list of labels associated with the image."""

    @abc.abstractmethod
    def add_base_metadata(self) -> None:
        """Add format-specific metadata to the root group."""

    @abc.abstractmethod
    def create_zarr_arrays(self):
        """Generate empty arrays inside the root group."""

    @abc.abstractmethod
    def read_binary_data(self) -> None:
        """Fill in the arrays with data from the source file."""

    def run(self) -> None:
        """Primary method to add metadata, create arrays, and read binary data."""
        self.add_base_metadata()
        self.create_zarr_arrays()
        self.read_binary_data()
