"""Experiment logging utilities for PanCirc-Fungi."""

import logging
import os
import sys
from datetime import datetime


def setup_logging(log_dir: str = "logs", name: str = "pancirc",
                  level: int = logging.INFO):
    """Set up logging to file and stdout.

    Configures BOTH the named logger AND the root logger, so that
    sub-module loggers (e.g. ``train.trainer``) inherit handlers
    and their messages appear in both the log file and stdout.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    # Configure root logger so sub-module loggers inherit handlers
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(level)

    # Named logger (may be the root if name == "root")
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Only add handlers if this logger doesn't already have them
    if logger.handlers:
        return logger

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(fh)
    root.addHandler(fh)

    # Stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))
    logger.addHandler(sh)
    root.addHandler(sh)

    return logger


class AverageMeter:
    """Track running averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
