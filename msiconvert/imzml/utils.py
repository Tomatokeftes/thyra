from typing import Optional, Tuple
from pathlib import Path

def get_imzml_pair(path: Path, imzml_filename: str, ibd_filename: str) -> Optional[Tuple[Path, Path]]:
    """Get the exact pair of '.imzml' and '.ibd' files by filename in the specified directory,
    verifying that both exist.

    Args:
        path (Path): Directory containing both files.
        imzml_filename (str): Name of the .imzML file to look for.
        ibd_filename (str): Name of the .ibd file to look for.

    Returns:
        Optional[Tuple[Path, Path]]: The pair of files, or None if no match is found.
    """
    imzml_path = path / imzml_filename
    ibd_path = path / ibd_filename

    # Verify both files exist
    if imzml_path.exists() and ibd_path.exists():
        return imzml_path, ibd_path
    return None  # Return None if either file is missing
