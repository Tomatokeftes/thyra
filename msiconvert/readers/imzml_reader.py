# msiconvert/readers/imzml_reader.py
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from pyimzml.ImzMLParser import ImzMLParser  # type: ignore
from tqdm import tqdm

from ..core.base_extractor import MetadataExtractor
from ..core.base_reader import BaseMSIReader
from ..core.registry import register_reader
from ..metadata.extractors.imzml_extractor import ImzMLMetadataExtractor


@register_reader("imzml")
class ImzMLReader(BaseMSIReader):
    """Reader for imzML format files with optimizations for performance."""

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 50,
        cache_coordinates: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize an ImzML reader.

        Args:
            data_path: Path to the imzML file
            batch_size: Default batch size for spectrum iteration
            cache_coordinates: Whether to cache coordinates upfront
            **kwargs: Additional arguments
        """
        super().__init__(data_path, **kwargs)
        self.filepath: Optional[Union[str, Path]] = data_path
        self.batch_size: int = batch_size
        self.cache_coordinates: bool = cache_coordinates
        self.parser: Optional[ImzMLParser] = None
        self.ibd_file: Optional[Any] = None
        self.imzml_path: Optional[Path] = None
        self.ibd_path: Optional[Path] = None
        self.is_continuous: bool = False
        self.is_processed: bool = False

        # Parser initialization flag for lazy loading
        self._parser_initialized: bool = False

        # Cached properties
        self._common_mass_axis: Optional[NDArray[np.float64]] = None
        self._coordinates_cache: Dict[int, Tuple[int, int, int]] = {}

        # Store path but don't initialize parser yet - wait for first use
        if data_path is not None:
            self.filepath = data_path

    def _ensure_parser_initialized(self) -> None:
        """Guarantee parser is initialized exactly once."""
        if not self._parser_initialized:
            if self.filepath is None:
                raise ValueError("No file path provided for parser initialization")
            self._initialize_parser(self.filepath)
            self._parser_initialized = True

    def _initialize_parser(self, imzml_path: Union[str, Path]) -> None:
        """
        Initialize the ImzML parser with the given path.

        Args:
            imzml_path: Path to the imzML file to parse

        Raises:
            ValueError: If the corresponding .ibd file is not found or metadata parsing fails
            Exception: If parser initialization fails
        """
        if isinstance(imzml_path, str):
            imzml_path = Path(imzml_path)

        self.imzml_path = imzml_path
        self.ibd_path = imzml_path.with_suffix(".ibd")

        if not self.ibd_path.exists():
            raise ValueError(f"Corresponding .ibd file not found for {imzml_path}")

        # Open the .ibd file for reading
        self.ibd_file = open(self.ibd_path, mode="rb")

        # Initialize the parser
        logging.info(f"Initializing ImzML parser for {imzml_path}")
        try:
            self.parser = ImzMLParser(
                filename=str(imzml_path), parse_lib="lxml", ibd_file=self.ibd_file
            )
        except Exception as e:
            if self.ibd_file:
                self.ibd_file.close()
            logging.error(f"Failed to initialize ImzML parser: {e}")
            raise

        if self.parser.metadata is None:
            raise ValueError("Failed to parse metadata from imzML file.")

        # Determine file mode
        self.is_continuous = "continuous" in self.parser.metadata.file_description.param_by_name  # type: ignore
        self.is_processed = "processed" in self.parser.metadata.file_description.param_by_name  # type: ignore

        if self.is_continuous == self.is_processed:
            raise ValueError(
                "Invalid file mode, expected either 'continuous' or 'processed'."
            )

        # Cache coordinates if requested
        if self.cache_coordinates:
            self._cache_all_coordinates()

    def _cache_all_coordinates(self) -> None:
        """
        Cache all coordinates for faster access.

        Converts 1-based coordinates from imzML to 0-based coordinates for internal use.
        """
        # Parser should already be initialized when this is called from _initialize_parser

        logging.info("Caching all coordinates for faster access")
        self._coordinates_cache = {}

        for idx, (x, y, z) in enumerate(self.parser.coordinates):  # type: ignore
            self._coordinates_cache[idx] = (x - 1, y - 1, z - 1 if z > 0 else 0)

        logging.info(f"Cached {len(self._coordinates_cache)} coordinates")

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create ImzML metadata extractor."""
        self._ensure_parser_initialized()

        if not self.imzml_path:
            raise ValueError("ImzML path not available")

        return ImzMLMetadataExtractor(self.parser, self.imzml_path)

    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """
        Return the common mass axis composed of all unique m/z values.

        For continuous mode, returns the m/z values from the first spectrum.
        For processed mode, collects all unique m/z values across spectra.

        Returns:
            NDArray[np.float64]: Array of m/z values in ascending order

        Raises:
            ValueError: If the common mass axis cannot be created
        """
        self._ensure_parser_initialized()

        if self._common_mass_axis is None:
            # We know parser is not None at this point
            parser = cast(ImzMLParser, self.parser)

            if self.is_continuous:
                logging.info("Using m/z values from first spectrum (continuous mode)")
                spectrum_data = parser.getspectrum(0)  # type: ignore
                if spectrum_data is None or len(spectrum_data) < 1:  # type: ignore
                    raise ValueError("Could not get first spectrum")

                mzs = spectrum_data[0]
                if mzs.size == 0:
                    raise ValueError("First spectrum contains no m/z values")

                self._common_mass_axis = mzs
            else:
                # For processed data, collect unique m/z values across spectra
                logging.info(
                    "Building common mass axis from all unique m/z values (processed mode)"
                )

                total_spectra = len(parser.coordinates)  # type: ignore

                all_mzs: List[NDArray[np.float64]] = []

                with tqdm(
                    total=total_spectra,
                    desc="Building common mass axis",
                    unit="spectrum",
                ) as pbar:
                    for idx in range(total_spectra):
                        try:
                            spectrum_data = parser.getspectrum(idx)  # type: ignore
                            if spectrum_data is None or len(spectrum_data) < 1:  # type: ignore
                                continue

                            mzs = spectrum_data[0]
                            if mzs.size > 0:
                                all_mzs.append(mzs)
                        except Exception as e:
                            logging.warning(f"Error getting spectrum {idx}: {e}")
                        pbar.update(1)

                if not all_mzs:
                    # No spectra found - raise exception instead of returning empty array
                    raise ValueError("No spectra found to build common mass axis")

                try:
                    combined_mzs = np.concatenate(all_mzs)
                    unique_mzs = np.unique(combined_mzs)

                    if unique_mzs.size == 0:
                        raise ValueError("Failed to extract any m/z values")

                    self._common_mass_axis = unique_mzs
                    logging.info(
                        f"Created common mass axis with {len(self._common_mass_axis)} unique m/z values"
                    )
                except Exception as e:
                    # Re-raise with more context
                    raise ValueError(f"Error creating common mass axis: {e}") from e

        # Return the common mass axis
        return self._common_mass_axis

    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """
        Iterate through spectra with progress monitoring and batch processing.

        Maps m/z values to the common mass axis using searchsorted for accurate
        representation in the output data structures.

        Args:
            batch_size: Number of spectra to process in each batch (None for default)

        Yields:
            Tuple containing:
                - Tuple[int, int, int]: Coordinates (x, y, z) - 0-based
                - NDArray[np.float64]: m/z values array
                - NDArray[np.float64]: Intensity values array

        Raises:
            ValueError: If parser is not initialized and no filepath is available
        """
        self._ensure_parser_initialized()

        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size

        # We know parser is not None at this point
        parser = cast(ImzMLParser, self.parser)

        total_spectra = len(parser.coordinates)  # type: ignore
        dimensions = self.get_essential_metadata().dimensions
        total_pixels = dimensions[0] * dimensions[1] * dimensions[2]

        # Log information about spectra vs pixels
        logging.info(
            f"Processing {total_spectra} spectra in a grid of {total_pixels} pixels"
        )

        # Process in batches
        with tqdm(
            total=total_spectra,
            desc="Reading spectra",
            unit="spectrum",
            disable=getattr(self, "_quiet_mode", False),
        ) as pbar:
            if batch_size <= 1:
                # Process one at a time
                for idx in range(total_spectra):
                    try:
                        # Get coordinates (using cached 0-based coordinates if available)
                        if idx in self._coordinates_cache:
                            coords = self._coordinates_cache[idx]
                        else:
                            x, y, z = parser.coordinates[idx]  # type: ignore
                            coords = cast(
                                Tuple[int, int, int],
                                (x - 1, y - 1, z - 1 if z > 0 else 0),
                            )

                        # Get spectrum data
                        mzs, intensities = parser.getspectrum(idx)  # type: ignore

                        if mzs.size > 0 and intensities.size > 0:
                            yield coords, mzs, intensities

                        pbar.update(1)
                    except Exception as e:
                        logging.warning(f"Error processing spectrum {idx}: {e}")
                        pbar.update(1)
            else:
                # Process in batches
                for batch_start in range(0, total_spectra, batch_size):
                    batch_end = min(batch_start + batch_size, total_spectra)
                    batch_size_actual = batch_end - batch_start

                    # Process each spectrum in the batch
                    for offset in range(batch_size_actual):
                        idx = batch_start + offset
                        try:
                            # Get coordinates (using cached 0-based coordinates if available)
                            if idx in self._coordinates_cache:
                                coords = self._coordinates_cache[idx]
                            else:
                                x, y, z = parser.coordinates[idx]  # type: ignore
                                coords = cast(
                                    Tuple[int, int, int],
                                    (x - 1, y - 1, z - 1 if z > 0 else 0),
                                )

                            # Get spectrum data
                            mzs, intensities = parser.getspectrum(idx)  # type: ignore

                            if mzs.size > 0 and intensities.size > 0:
                                yield coords, mzs, intensities

                            pbar.update(1)
                        except Exception as e:
                            logging.warning(f"Error processing spectrum {idx}: {e}")
                            pbar.update(1)

    def read(self) -> Dict[str, Any]:
        """
        Read the entire imzML file and return a structured data dictionary.

        Returns:
            Dict containing:
                - mzs: NDArray[np.float64] - common m/z values array
                - intensities: NDArray[np.float64] - array of intensity arrays
                - coordinates: List[Tuple[int, int, int]] - list of (x,y,z) coordinates
                - width: int - number of pixels in x dimension
                - height: int - number of pixels in y dimension
                - depth: int - number of pixels in z dimension

        Raises:
            ValueError: If parser is not initialized and no filepath is available
        """
        self._ensure_parser_initialized()

        # Get common mass axis
        mzs = self.get_common_mass_axis()

        # Get dimensions
        width, height, depth = self.get_essential_metadata().dimensions

        # Collect all spectra
        coordinates: List[Tuple[int, int, int]] = []
        intensities: List[NDArray[np.float64]] = []

        # Iterate through all spectra
        for coords, spectrum_mzs, spectrum_intensities in self.iter_spectra():
            coordinates.append(coords)

            # Convert sparse representation to full array
            full_spectrum = np.zeros(len(mzs), dtype=np.float64)

            # Find indices in the common mass axis using searchsorted
            indices = np.searchsorted(mzs, spectrum_mzs)

            # Ensure indices are within bounds
            valid_indices = indices < len(mzs)
            indices = indices[valid_indices]
            valid_intensities = spectrum_intensities[valid_indices]

            # Fill spectrum array
            full_spectrum[indices] = valid_intensities
            intensities.append(full_spectrum)

        return {
            "mzs": mzs,
            "intensities": np.array(intensities, dtype=np.float64),
            "coordinates": coordinates,
            "width": width,
            "height": height,
            "depth": depth,
        }

    def close(self) -> None:
        """Close all open file handles."""
        if hasattr(self, "ibd_file") and self.ibd_file is not None:
            self.ibd_file.close()
            self.ibd_file = None

        if hasattr(self, "parser") and self.parser is not None:
            if hasattr(self.parser, "m") and self.parser.m is not None:
                self.parser.m.close()  # type: ignore
            self.parser = None

    @property
    def n_spectra(self) -> int:
        """
        Return the total number of spectra in the dataset.

        Returns:
            Total number of spectra (efficient implementation using parser)
        """
        self._ensure_parser_initialized()

        # Use parser coordinates which is efficient
        parser = cast(ImzMLParser, self.parser)
        return len(parser.coordinates)  # type: ignore
