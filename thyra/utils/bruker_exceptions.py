"""
Custom exceptions for the Bruker reader.

This module provides specific exception types for better error handling
and debugging throughout the reader implementation.
"""


class BrukerReaderError(Exception):
    """Base exception for all Bruker reader errors."""

    pass


class SDKError(BrukerReaderError):
    """Exception raised when there are issues with the Bruker SDK."""

    def __init__(self, message: str, sdk_error_code: int = None):
        super().__init__(message)
        self.sdk_error_code = sdk_error_code


class DataError(BrukerReaderError):
    """Exception raised when there are issues with the data itself."""

    pass


class ConfigurationError(BrukerReaderError):
    """Exception raised when there are configuration or setup issues."""

    pass


class MemoryError(BrukerReaderError):
    """Exception raised when there are memory-related issues."""

    pass


class FileFormatError(BrukerReaderError):
    """Exception raised when the file format is not supported or corrupted."""

    pass
