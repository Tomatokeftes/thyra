"""Configuration constants for MSIConverter.

This module centralizes all configuration values, magic numbers, and
default settings used throughout the codebase to improve maintainability
and allow easier tuning for different use cases.
"""

# Memory and performance settings
DEFAULT_BUFFER_SIZE = 100000  # Default buffer size for processing spectra
DEFAULT_BATCH_SIZE_IMZML = 50  # Default batch size for ImzML processing
DEFAULT_BATCH_SIZE_BRUKER = 100  # Default batch size for Bruker processing

# Memory constraints
MB_TO_BYTES = 1024 * 1024
BYTES_PER_FLOAT64 = 8
BYTES_PER_FLOAT32 = 4
BYTES_PER_UINT32 = 4
DEFAULT_MEMORY_LIMIT_MB = 1024  # Default memory limit in MB
LOG_FILE_MAX_SIZE_MB = 10  # Max log file size before rotation
LOG_BACKUP_COUNT = 5

# Batch processing limits
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE_BRUKER = 1000
ADAPTIVE_BATCH_ADJUSTMENT = 10  # Amount to adjust batch size

# Performance thresholds
FAST_PROCESSING_THRESHOLD_SEC = 0.5  # Under this is considered fast
SLOW_PROCESSING_THRESHOLD_SEC = 2.0  # Above this is considered slow

# Coordinate cache settings
COORDINATE_CACHE_BATCH_SIZE = 100
BUFFER_POOL_SIZE = 20

# SDK buffer settings
INITIAL_BUFFER_SIZE = 1024  # Initial buffer size for SDK operations

# Mass axis building
MASS_AXIS_BATCH_SIZE = 50  # Batch size for mass axis construction

# Memory pool settings
MAX_BUFFERS_PER_SIZE = 10  # Maximum number of buffers to keep per size

# Pixel size detection
PIXEL_SIZE_TOLERANCE = 0.01  # Tolerance for pixel size comparison (micrometers)

# File size estimation (rough)
ESTIMATED_BYTES_PER_SPECTRUM_POINT = 4  # For float32 values

# Progress reporting
DEFAULT_PROGRESS_UPDATE_INTERVAL = 1.0  # Seconds between progress updates
