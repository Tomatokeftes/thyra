import logging
import logging.handlers
import sys

from ..config import LOG_BACKUP_COUNT, LOG_FILE_MAX_SIZE_MB, MB_TO_BYTES


def setup_logging(log_level=logging.INFO, log_file=None):
    """Set up logging for the application.

    Args:
        log_level (int): The minimum logging level to display.
        log_file (str): Path to the log file. If None, logs are not saved to a file.
    """
    # Get the root logger
    logger = logging.getLogger("thyra")
    logger.setLevel(log_level)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    # Remove all existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a file handler if a log file is specified
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_FILE_MAX_SIZE_MB * MB_TO_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging configured")
