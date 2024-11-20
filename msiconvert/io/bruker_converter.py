from typing import List, Tuple
from pathlib import Path
import numpy as np
import zarr
from .base_converter import BaseMSIConverter
from .registry import register_converter
from ..utils.zarr_manager import ZarrManager
import sys
import sqlite3
from ctypes import *
from tqdm import tqdm

SHAPE = Tuple[int, int, int, int]

class BrukerParser:
    """Class to parse Bruker TSF data files."""

    def __init__(self, analysis_directory: Path, use_recalibrated_state: bool = False):
        self.analysis_directory = analysis_directory
        self.use_recalibrated_state = use_recalibrated_state
        self.handle = None
        self.conn = None
        self.line_buffer_size = 1024  # May grow in read methods
        self.profile_buffer_size = 1024  # May grow in read methods

        self._load_dll()
        self._open_data()
        self._connect_database()
        self._load_metadata()

    @property
    def intensity_dtype(self):
        return np.float32 
    
    @property
    def mz_dtype(self):
        return np.float64

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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
        self.dll.tsf_open.argtypes = [c_char_p, c_uint32]
        self.dll.tsf_open.restype = c_uint64
        self.dll.tsf_close.argtypes = [c_uint64]
        self.dll.tsf_close.restype = None
        self.dll.tsf_get_last_error_string.argtypes = [c_char_p, c_uint32]
        self.dll.tsf_get_last_error_string.restype = c_uint32
        self.dll.tsf_read_line_spectrum.argtypes = [
            c_uint64, c_int64, POINTER(c_double), POINTER(c_float), c_uint32
        ]
        self.dll.tsf_read_line_spectrum.restype = c_uint32
        self.dll.tsf_index_to_mz.argtypes = [
            c_uint64, c_int64, POINTER(c_double), POINTER(c_double), c_uint32
        ]
        self.dll.tsf_index_to_mz.restype = c_uint32
        # Define other necessary functions similarly...

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

    def close(self):
        """Close the data handle and database connection."""
        if self.handle:
            self.dll.tsf_close(self.handle)
            self.handle = None
        if self.conn:
            self.conn.close()
            self.conn = None

    def get_max_spectrum_length(self) -> int:
        """Compute the maximum spectrum length."""
        max_length = 0
        frame_count = self.get_frame_count()
        for frame_id in range(1, frame_count + 1):
            indices, _ = self.read_line_spectrum(frame_id)
            length = len(indices)
            if length > max_length:
                max_length = length
        return max_length

    def index_to_mz(self, frame_id: int, indices: np.ndarray) -> np.ndarray:
        """Convert indices to m/z values."""
        cnt = len(indices)
        mzs = np.empty(cnt, dtype=np.float64)
        success = self.dll.tsf_index_to_mz(
            self.handle,
            frame_id,
            indices.ctypes.data_as(POINTER(c_double)),
            mzs.ctypes.data_as(POINTER(c_double)),
            cnt,
        )
        if success == 0:
            self._throw_last_error()
        return mzs

    def read_line_spectrum(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read the line spectrum for a given frame."""
        while True:
            cnt = int(self.line_buffer_size)
            index_buf = np.empty(cnt, dtype=np.float64)
            intensity_buf = np.empty(cnt, dtype=np.float32)

            required_len = self.dll.tsf_read_line_spectrum(
                self.handle,
                frame_id,
                index_buf.ctypes.data_as(POINTER(c_double)),
                intensity_buf.ctypes.data_as(POINTER(c_float)),
                cnt,
            )

            if required_len == 0:
                # Empty spectrum
                return np.array([]), np.array([])

            if required_len > self.line_buffer_size:
                if required_len > 16777216:
                    raise RuntimeError("Maximum expected frame size exceeded.")
                self.line_buffer_size = required_len
            else:
                break

        return index_buf[:required_len], intensity_buf[:required_len]

    def get_spectrum(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the m/z and intensity arrays for a given frame."""
        indices, intensities = self.read_line_spectrum(frame_id)
        if len(indices) == 0:
            return np.array([]), np.array([])
        mzs = self.index_to_mz(frame_id, indices)
        return mzs, intensities

    def get_frame_positions(self) -> np.ndarray:
        """Get the (x, y) positions of all frames."""
        return self.frame_positions

    def get_frame_count(self) -> int:
        """Get the total number of frames."""
        return self.frame_count
    
@register_converter('bruker')
class BrukerConverter(BaseMSIConverter):
    def __init__(self, root: zarr.Group, name: str, input_path: Path):
        super().__init__(root, name)
        self.input_path = input_path
        self.reader = BrukerParser(input_path)
        self.zarr_manager = None  # Will initialize in create_zarr_arrays()

    def get_labels(self) -> List[str]:
        return ['mzs/0', 'lengths/0']

    def get_intensity_shape(self) -> SHAPE:
        # Get maximum spectrum length and image dimensions
        max_mz_length = self.reader.get_max_spectrum_length()
        y_max = int(np.max(self.reader.frame_positions[:, 1])) + 1
        x_max = int(np.max(self.reader.frame_positions[:, 0])) + 1
        return (max_mz_length, 1, y_max, x_max)

    def get_mz_shape(self) -> SHAPE:
        return self.get_intensity_shape()

    def get_lengths_shape(self) -> SHAPE:
        y_max = int(np.max(self.reader.frame_positions[:, 1])) + 1
        x_max = int(np.max(self.reader.frame_positions[:, 0])) + 1
        return (1, 1, y_max, x_max)

    def add_base_metadata(self) -> None:
        self.root.attrs['multiscales'] = [{
            'version': '0.4',
            'name': self.name,
            'datasets': [{'path': '0'}],
            'axes': ['c', 'z', 'y', 'x'],
            'type': 'none',
        }]
        self.root.attrs['bruker'] = {
            'source': str(self.input_path),
            'frame_count': self.reader.get_frame_count(),
        }
        self.root.create_group('labels').attrs['labels'] = self.get_labels()

    def create_zarr_arrays(self):
        # Initialize ZarrManager
        self.zarr_manager = ZarrManager(self.root, self.reader)
        self.zarr_manager.create_arrays(
            self.get_intensity_shape,
            self.get_mz_shape,
            self.get_lengths_shape
        )

    def read_binary_data(self) -> None:
        with self.zarr_manager.temporary_arrays():
            frame_positions = self.reader.get_frame_positions()
            frame_count = self.reader.get_frame_count()

            # Initialize the tqdm progress bar
            with tqdm(total=frame_count, desc="Processing frames", unit="frame") as pbar:
                for frame_id in range(1, frame_count + 1):
                    mzs, intensities = self.reader.get_spectrum(frame_id)
                    if mzs.size == 0:
                        pbar.update(1)  # Update the progress bar even for skipped frames
                        continue

                    x_pos = int(frame_positions[frame_id - 1][0])
                    y_pos = int(frame_positions[frame_id - 1][1])

                    length = len(mzs)
                    self.zarr_manager.lengths[0, 0, y_pos, x_pos] = length
                    self.zarr_manager.fast_mzs[:length, 0, y_pos, x_pos] = mzs
                    self.zarr_manager.fast_intensities[:length, 0, y_pos, x_pos] = intensities

                    # Update the progress bar
                    pbar.update(1)

            self.zarr_manager.copy_to_main_arrays()

    def run(self) -> None:
        try:
            self.add_base_metadata()
            self.create_zarr_arrays()
            self.read_binary_data()
        finally:
            self.reader.close()
