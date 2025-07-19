import logging
import logging.handlers
import sys


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging for the application.

    Args:
        log_level (int): The minimum logging level to display.
        log_file (str): Path to the log file. If None, logs are not saved to a file.
    """
    # Get the root logger
    logger = logging.getLogger("msiconvert")
    logger.setLevel(log_level)

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
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logging configured")
