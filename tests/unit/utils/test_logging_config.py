import pytest
import logging
from pathlib import Path
from msiconvert.utils.logging_config import setup_logging

@pytest.fixture
def logger():
    # Ensure the logger is clean for each test
    logger = logging.getLogger("msiconvert")
    logger.handlers.clear()
    return logger

def test_setup_logging_console(logger):
    """Test that console logging is set up correctly."""
    setup_logging(log_level=logging.DEBUG)
    
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.DEBUG

def test_setup_logging_file(logger, tmp_path):
    """Test that file logging is set up correctly when a log file is provided."""
    log_file = tmp_path / "test.log"
    setup_logging(log_level=logging.INFO, log_file=str(log_file))
    
    assert len(logger.handlers) == 2
    
    file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            file_handler = handler
            break
    
    assert file_handler is not None
    assert file_handler.level == logging.INFO
    assert Path(file_handler.baseFilename).name == "test.log"

    # Test that a message is written to the log file
    test_message = "This is a test log message."
    logger.info(test_message)
    
    # Close the handler to ensure the message is flushed to the file
    file_handler.close()
    
    with open(log_file, 'r') as f:
        log_content = f.read()
        assert test_message in log_content

def test_log_level_filtering(logger, tmp_path):
    """Test that log messages are filtered based on the configured log level."""
    log_file = tmp_path / "test_filtering.log"
    setup_logging(log_level=logging.WARNING, log_file=str(log_file))
    
    debug_message = "This is a debug message."
    warning_message = "This is a warning message."
    
    logger.debug(debug_message)
    logger.warning(warning_message)

    # Close file handler to ensure logs are written
    for handler in logger.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler):
            handler.close()

    with open(log_file, 'r') as f:
        log_content = f.read()
        assert debug_message not in log_content
        assert warning_message in log_content
