"""
Platform detection and SDK library path discovery.

This module provides robust platform detection and automatic discovery of
Bruker SDK libraries across different operating systems and installation paths.
"""

import logging
import platform
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class PlatformDetector:
    """Detects platform and provides appropriate SDK paths."""

    @staticmethod
    def get_platform() -> str:
        """
        Get normalized platform identifier.

        Returns:
            Platform string: 'windows', 'linux', or 'macos'
        """
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        return system

    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows."""
        return PlatformDetector.get_platform() == "windows"

    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux."""
        return PlatformDetector.get_platform() == "linux"

    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS."""
        return PlatformDetector.get_platform() == "macos"


def get_dll_paths(data_directory: Optional[Path] = None) -> List[Path]:
    """
    Get list of potential DLL/SO library paths for the current platform.

    This combines the best path detection logic from timsconvert and imzy
    implementations to provide comprehensive coverage.

    Args:
        data_directory: Optional data directory to check for local libraries

    Returns:
        List of Path objects representing potential library locations
    """
    paths = []
    platform_name = PlatformDetector.get_platform()

    if platform_name == "windows":
        # Windows DLL paths
        dll_name = "timsdata.dll"

        # Standard installation paths
        paths.extend(
            [
                Path("C:/Program Files/Bruker/timsTOF/sdk") / dll_name,
                Path("C:/Program Files (x86)/Bruker/timsTOF/sdk") / dll_name,
                Path("C:/Bruker/sdk") / dll_name,
                Path("C:/Bruker/timsdata") / dll_name,
            ]
        )

        # User-specific paths (from timsconvert)
        user_paths = [
            Path(r"C:\Users\P70078823\Desktop\MSIConverter") / dll_name,
            Path.home() / "Desktop" / "MSIConverter" / dll_name,
            Path.home() / "Downloads" / dll_name,
            Path.home() / "Documents" / "Bruker" / dll_name,
        ]
        paths.extend(user_paths)

        # Local data directory
        if data_directory:
            paths.append(data_directory.parent / dll_name)
            paths.append(data_directory / dll_name)

        # Current working directory and PATH
        paths.extend(
            [
                Path.cwd() / dll_name,
                Path(dll_name),  # Rely on system PATH
            ]
        )

    elif platform_name == "linux":
        # Linux SO paths
        so_name = "libtimsdata.so"

        # Standard library paths
        paths.extend(
            [
                Path("/usr/lib") / so_name,
                Path("/usr/local/lib") / so_name,
                Path("/opt/bruker/lib") / so_name,
                Path("/usr/lib/x86_64-linux-gnu") / so_name,
            ]
        )

        # Local data directory
        if data_directory:
            paths.append(data_directory.parent / so_name)
            paths.append(data_directory / so_name)

        # Current working directory and LD_LIBRARY_PATH
        paths.extend(
            [
                Path.cwd() / so_name,
                Path(so_name),  # Rely on LD_LIBRARY_PATH
            ]
        )

    elif platform_name == "macos":
        # macOS dylib paths (limited support)
        dylib_name = "libtimsdata.dylib"

        paths.extend(
            [
                Path("/usr/local/lib") / dylib_name,
                Path("/opt/bruker/lib") / dylib_name,
            ]
        )

        if data_directory:
            paths.append(data_directory.parent / dylib_name)
            paths.append(data_directory / dylib_name)

        paths.extend(
            [
                Path.cwd() / dylib_name,
                Path(dylib_name),
            ]
        )

    # Filter to only existing paths and log findings
    existing_paths = []
    for path in paths:
        if path.exists():
            existing_paths.append(path)
            logger.debug(f"Found potential SDK library: {path}")
        else:
            logger.debug(f"SDK library not found: {path}")

    if existing_paths:
        logger.info(f"Found {len(existing_paths)} potential SDK libraries")
    else:
        logger.warning("No SDK libraries found in standard locations")

    return existing_paths


def get_library_name() -> str:
    """
    Get the appropriate library name for the current platform.

    Returns:
        Library filename (e.g., 'timsdata.dll', 'libtimsdata.so')
    """
    platform_name = PlatformDetector.get_platform()

    if platform_name == "windows":
        return "timsdata.dll"
    elif platform_name == "linux":
        return "libtimsdata.so"
    elif platform_name == "macos":
        return "libtimsdata.dylib"
    else:
        raise RuntimeError(f"Unsupported platform: {platform_name}")


def validate_library_path(library_path: Path) -> bool:
    """
    Validate that a library path is accessible and loadable.

    Args:
        library_path: Path to the library file

    Returns:
        True if the library appears to be valid, False otherwise
    """
    if not library_path.exists():
        return False

    if not library_path.is_file():
        return False

    # Basic size check (libraries should be substantial)
    try:
        size = library_path.stat().st_size
        if size < 1024:  # Less than 1KB is suspicious
            logger.warning(
                f"Library file seems too small: {library_path} ({size} bytes)"
            )
            return False
    except OSError:
        return False

    return True
