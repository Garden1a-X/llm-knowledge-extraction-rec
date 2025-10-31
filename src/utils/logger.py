"""
Logging utilities using loguru.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days",
    **kwargs
):
    """
    Setup logger with file and console handlers.

    Args:
        log_file: Path to log file (if None, only console logging)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        rotation: When to rotate log file (e.g., "100 MB", "1 day")
        retention: How long to keep old logs
        **kwargs: Additional arguments to pass to logger.add()
    """
    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
        **kwargs
    )

    # Add file handler if log_file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            **kwargs
        )

        logger.info(f"Logger initialized. Log file: {log_file}")

    return logger


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance.

    Args:
        name: Name of the logger (typically __name__)

    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Example usage in other modules:
# from src.utils.logger import get_logger
# logger = get_logger(__name__)
# logger.info("This is an info message")
