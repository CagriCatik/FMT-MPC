from __future__ import annotations

import logging
from pathlib import Path

from src.common.logging_utils import configure_logging


def test_configure_logging_creates_timestamped_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_path = configure_logging(log_dir)
    logger = logging.getLogger("test.logging")
    logger.info("logging smoke test")
    logging.shutdown()
    assert log_path.exists()
    content = log_path.read_text()
    assert "logging smoke test" in content
