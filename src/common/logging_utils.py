"""Utility helpers for configuring deterministic application logging.

This module centralises logging setup so every entry-point configures the
Python ``logging`` module in a consistent, explicit way.  The helper exposes a
single ``configure_logging`` function that attaches both a console handler and a
file handler with identical formatting.  The file handler writes timestamped
``.log`` files into ``output/logs`` (or a user-provided directory) to ensure
that terminal output is always captured for later inspection.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(output_directory: Path | None = None) -> Path:
    """Configure root logging handlers and return the created log file path.

    Parameters
    ----------
    output_directory:
        Optional directory in which the timestamped log file should be
        created.  When ``None`` (the default) the helper writes to
        ``output/logs`` inside the project root.

    Returns
    -------
    Path
        The fully-qualified path to the log file receiving mirrored console
        output for the current run.
    """

    log_dir = Path(output_directory) if output_directory else Path("output/logs")
    if not log_dir.is_absolute():
        log_dir = Path.cwd() / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"run_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Remove any pre-existing handlers so repeated invocations of the CLI do
    # not duplicate log entries.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.debug("Configured logging; writing mirrored output to %s", log_path)

    return log_path

