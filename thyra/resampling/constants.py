"""Constants for resampling and instrument detection."""


class ImzMLAccessions:
    """PSI-MS and imzML controlled vocabulary accession codes."""

    # Spectrum type (MS ontology)
    CENTROID_SPECTRUM = "MS:1000127"
    PROFILE_SPECTRUM = "MS:1000128"

    # Binary data type (imzML ontology)
    CONTINUOUS_BINARY = "IMS:1000030"
    PROCESSED_BINARY = "IMS:1000031"

    # Software identifiers
    SCILS_LAB = "MS:1002384"


class Thresholds:
    """Threshold values for data classification."""

    # Peak density threshold for detecting profile data
    # Profile data typically has >5000 points per spectrum
    PROFILE_PEAK_DENSITY = 5000

    # Dataset size threshold for streaming converter (GB)
    STREAMING_SIZE_GB = 10

    # Dataset size threshold for PCS method in streaming converter (GB)
    PCS_SIZE_GB = 30


class SpectrumType:
    """Spectrum type string constants."""

    CENTROID = "centroid spectrum"
    PROFILE = "profile spectrum"


class BinaryDataType:
    """Binary data type string constants."""

    CONTINUOUS = "continuous"
    PROCESSED = "processed"
