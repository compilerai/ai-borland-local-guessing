"""
logging_utils.py

Drop-in logging setup that:
  - Prints to terminal (with tqdm compatibility)
  - Writes timestamped lines to logs/output_<tag>.txt
  - Integrates with Python's standard logging module
  - Works safely with HuggingFace / tqdm progress bars

Usage:
    from logging_utils import setup_logging

    setup_logging(tag="run_1")          # logs/output_run_1.txt
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Training started")
"""

import logging
import os
import sys
from datetime import datetime

# File handler that timestamps every line
class _TimestampedFileHandler(logging.FileHandler):
    """
    A FileHandler whose formatter prefixes every record with [HH:MM:SS].
    Kept separate from the terminal formatter so the terminal can use its
    own format string.
    """

    _FMT = "[%(asctime)s] %(levelname)-8s | %(name)s | %(message)s"
    _DATEFMT = "%H:%M:%S"

    def __init__(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        super().__init__(filepath, mode="w", encoding="utf-8", delay=False)
        self.setFormatter(logging.Formatter(self._FMT, datefmt=self._DATEFMT))

# Tqdm-safe stream handler
class _TqdmStreamHandler(logging.StreamHandler):
    """
    Writes log records via tqdm.write() when tqdm is available so that
    progress bars are not clobbered by log lines.
    Falls back to plain stderr if tqdm is not installed.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            import tqdm
            tqdm.tqdm.write(self.format(record), file=sys.stderr)
        except ImportError:
            super().emit(record)

# Public API
def setup_logging(
    tag: str = "run",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> str:
    """
    Configure the root logger with:
      - A tqdm-safe console handler (console_level, default INFO)
      - A timestamped file handler   (file_level,   default DEBUG)

    Parameters
    ----------
    tag           : Suffix for the log filename  →  logs/output_<tag>.txt
    log_dir       : Directory to write log files into
    console_level : Minimum level printed to terminal
    file_level    : Minimum level written to file (DEBUG captures everything)

    Returns
    -------
    str  : Absolute path of the log file created
    """
    log_path = os.path.join(log_dir, f"output_{tag}.txt")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers filter individually

    # Avoid duplicate handlers if setup_logging is called more than once
    root.handlers.clear()

    # --- Console handler ---
    console_handler = _TqdmStreamHandler(stream=sys.stderr)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(console_handler)

    # --- File handler ---
    file_handler = _TimestampedFileHandler(log_path)
    file_handler.setLevel(file_level)
    root.addHandler(file_handler)

    # Banner written to both sinks via the root logger
    banner_logger = logging.getLogger("setup")
    banner_logger.info("=" * 70)
    banner_logger.info(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    banner_logger.info(f"Log file : {os.path.abspath(log_path)}")
    banner_logger.info("=" * 70)

    return os.path.abspath(log_path)