"""Logging utilities."""
from __future__ import annotations

import logging
import os
import sys


def setup_logger(
    name: str = "enfm",
    out_dir: str = "./runs",
    filename: str = "train.log",
) -> logging.Logger:
    """Configures and returns a logger that outputs to both console and a log file.

    Args:
        name: Name of the logger.
        out_dir: Directory where the log file will be saved.
        filename: Name of the log file.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if the logger is setup multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Stream handler for console stdout
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler for log file
        log_file_path = os.path.join(out_dir, filename)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
