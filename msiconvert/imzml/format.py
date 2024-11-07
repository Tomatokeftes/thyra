from functools import cached_property
from pathlib import Path

from msiconvert.imzml.checker import ImzMLChecker
from msiconvert.imzml.convertor import ImzMLToZarrConvertor
from msiconvert.imzml.parser import ImzMLParser

class NotImplementedClass:
    """Class that does nothing and raises NotImplementedError on access."""
    __slots__ = ()

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the NotImplementedClass.

        Parameters:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        pass

    def __getattribute__(self, __name: str):
        """
        Override to raise NotImplementedError on attribute access.

        Parameters:
        __name (str): The name of the attribute being accessed.

        Raises:
        NotImplementedError: Always raised to indicate the class is not implemented.
        """
        raise NotImplementedError("This class is not implemented.")

    def __setattr__(self, __name: str, __value) -> None:
        """
        Override to raise NotImplementedError on attribute setting.

        Parameters:
        __name (str): The name of the attribute being set.
        __value: The value being assigned to the attribute.

        Raises:
        NotImplementedError: Always raised to indicate the class is not implemented.
        """
        raise NotImplementedError("This class is not implemented.")

class ImzMLFormat:
    """Independent Format definition for ImzML."""

    checker_class = ImzMLChecker
    parser_class = ImzMLParser
    reader_class = NotImplementedClass  # Placeholder for future implementation
    convertor_class = ImzMLToZarrConvertor
    histogram_reader_class = NotImplementedClass  # Placeholder for future implementation

    def __init__(self, path: Path) -> None:
        """
        Initialize the ImzMLFormat.

        Parameters:
        path (Path): The path to the ImzML file.
        """
        self._path = path
        self._enabled = True

    @classmethod
    def get_name(cls) -> str:
        """
        Return the name of the file format.

        Returns:
        str: The name of the file format.
        """
        return "ImzML"

    @classmethod
    def is_spatial(cls) -> bool:
        """
        Indicate if the format is spatial.

        Returns:
        bool: False, indicating the format is not spatial.
        """
        return False

    @classmethod
    def is_spectral(cls) -> bool:
        """
        Indicate if the format is spectral.

        Returns:
        bool: False, indicating the format is not spectral.
        """
        return False

    @classmethod
    def is_writable(cls) -> bool:
        """
        Indicate if the format is writable.

        Returns:
        bool: False, indicating the format is not writable.
        """
        return False

    @cached_property
    def need_conversion(self) -> bool:
        """
        Indicate whether the image needs conversion.

        Returns:
        bool: True, indicating the image needs conversion.
        """
        return True
