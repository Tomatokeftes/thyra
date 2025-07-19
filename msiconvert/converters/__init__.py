try:
    from . import spatialdata_converter
except (ImportError, NotImplementedError):
    # Skip if spatialdata dependencies not available or incompatible
    import logging
    logging.warning("SpatialData converter not available due to dependency issues")