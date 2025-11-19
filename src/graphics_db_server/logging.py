"""Logging configuration for graphics-db-server."""

import sys

from loguru import logger


def configure_logging(level="INFO", sink=sys.stderr):
    """
    Configures the Loguru logger for the library.

    This function removes the default Loguru handler and adds a new one with
    the specified parameters, providing a simple way to set up logging.

    Args:
        level (str, optional): The minimum logging level to output.
            Defaults to "INFO".
        sink (file-like object, optional): The destination for logs.
            Defaults to `sys.stderr`.
        format (str, optional): The Loguru format string for the log messages.
            Defaults to "{level: <9} {message}".

    Returns:
        The configured logger instance.
    """
    logger.remove()
    logger.add(sink, level=level)
    return logger
