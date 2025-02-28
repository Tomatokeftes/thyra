# msiconvert/readers/bruker_reader.py
import numpy as np
from typing import Dict, Any, Tuple, Generator, Optional
from pathlib import Path
import sqlite3
from ctypes import *
import sys

from ..core.base_reader import BaseMSIReader

class BrukerReader(BaseMSIReader):
    """Reader for Bruker TSF format files."""
    
    def __init__(self, analysis_directory: Path, use_recalibrated_state: bool = False):
        self.analysis_directory = analysis_directory
        self.use_recalibrated_state = use_recalibrated_state
        self.handle = None
        self.conn = None
        self.line_buffer_size = 1024  # May grow in read methods
        
        self._load_dll()
        self._open_data()
        self._connect_database()
        self._load_metadata()
        
        # Calculate common mass axis
        self._common_mass_axis = None
    
    def _load_dll(self):
        """Load the timsdata DLL or shared library."""
        if sys.platform.startswith("win32"):
            libname = r"C:\Users\tvisv\Downloads\MSIConverter\timsdata.dll"
        elif sys.platform.startswith("linux"):
            libname = "libtimsdata.so"
        else:
            raise Exception("Unsupported platform.")
        
        self.dll = cdll.LoadLibrary(libname)
        # Define argument and return types for DLL functions
        self._define_dll_functions()
    
    def _define_dll_functions(self):
        """Define the argument and return types for DLL functions."""
        # Implementation details...
    
    def _throw_last_error(self):
        """Retrieve and raise the last error from the DLL."""
        len_buf = self.dll.tsf_get_last_error_string(None, 0)
        buf = create_string_buffer(len_buf)
        self.dll.tsf_get_last_error_string(buf, len_buf)
        raise RuntimeError(buf.value.decode('utf-8'))
    
    def _open_data(self):
        """Open the TSF data file."""
        self.handle = self.dll.tsf_open(
            str(self.analysis_directory).encode('utf-8'),
            1 if self.use_recalibrated_state else 0,
        )
        if self.handle == 0:
            self._throw_last_error()
    
    def _connect_database(self):
        """Connect to the SQLite database within the TSF data directory."""
        db_path = self.analysis_directory / "analysis.tsf"
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        self.conn = sqlite3.connect(str(db_path))
    
    def _load_metadata(self):
        """Load necessary metadata from the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Frames")
        self.frame_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT XIndexPos, YIndexPos FROM MaldiFrameInfo")
        positions = cursor.fetchall()
        self.frame_positions = np.array(positions)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the Bruker dataset."""
        return {
            'source': str(self.analysis_directory),
            'frame_count': self.frame_count
        }
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return the dimensions of the Bruker dataset (x, y, z)."""
        x_max = int(np.max(self.frame_positions[:, 0])) + 1
        y_max = int(np.max(self.frame_positions[:, 1])) + 1
        z_max = 1  # Assume 2D data for now
        return (x_max, y_max, z_max)
    
    def get_common_mass_axis(self) -> np.ndarray:
        """Return the common mass axis for all spectra."""
        if self._common_mass_axis is None:
            # Collect all m/z values
            all_mz_arrays = []
            for frame_id in range(1, self.frame_count + 1):
                mzs, _ = self.get_spectrum(frame_id)
                if mzs.size > 0:
                    all_mz_arrays.append(mzs)
            
            # Concatenate and find unique values
            all_mzs = np.concatenate(all_mz_arrays)
            self._common_mass_axis = np.unique(all_mzs)
        
        return self._common_mass_axis
    
    def iter_spectra(self) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through spectra.
        
        Yields:
        -------
        Tuple containing:
            - Coordinates (x, y, z)
            - m/z values array
            - Intensity values array
        """
        for frame_id in range(1, self.frame_count + 1):
            mzs, intensities = self.get_spectrum(frame_id)
            if mzs.size > 0:
                x_pos = int(self.frame_positions[frame_id - 1][0])
                y_pos = int(self.frame_positions[frame_id - 1][1])
                z_pos = 0  # Assume 2D data for now
                
                yield ((x_pos, y_pos, z_pos), mzs, intensities)
    
    def get_spectrum(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the m/z and intensity arrays for a given frame."""
        indices, intensities = self.read_line_spectrum(frame_id)
        if len(indices) == 0:
            return np.array([]), np.array([])
        mzs = self.index_to_mz(frame_id, indices)
        return mzs, intensities
    
    def read_line_spectrum(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read the line spectrum for a given frame."""
        # Implementation details...
    
    def index_to_mz(self, frame_id: int, indices: np.ndarray) -> np.ndarray:
        """Convert indices to m/z values."""
        # Implementation details...
    
    def close(self) -> None:
        """Close the data handle and database connection."""
        if self.handle:
            self.dll.tsf_close(self.handle)
            self.handle = None
        if self.conn:
            self.conn.close()
            self.conn = None