"""
Logger Utility
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: str = None,
    rotation: str = "10 MB",
    retention: str = "7 days"
):
    """
    Setup logger with file and console output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None for console only)
        rotation: Log rotation size
        retention: Log retention period
    """
    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>"
    )

    # Add file logger if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                   "{name}:{function}:{line} | {message}",
            rotation=rotation,
            retention=retention,
            compression="zip"
        )

    logger.info(f"Logger initialized with level: {log_level}")


def get_logger(name: str = None):
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger
