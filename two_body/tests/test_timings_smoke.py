from __future__ import annotations

import csv
import time
from pathlib import Path

from perf_timings import (
    CSV_HEADER,
    configure_global_logger,
    shutdown_logger,
    time_block,
    time_section,
)


def test_time_block_and_decorator(tmp_path: Path) -> None:
    logger = configure_global_logger(enabled=True, base_dir=tmp_path, buffer_size=1)

    with time_block("batch_eval", epoch=1, batch_id=2, extra={"test": True}):
        time.sleep(0.0001)

    @time_section("fitness_eval", batch_id=2, individual_id=7)
    def _dummy() -> int:
        time.sleep(0.0001)
        return 5

    assert _dummy() == 5
    logger.flush()
    csv_path = logger.path
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    assert rows[0] == CSV_HEADER
    assert len(rows) >= 3  # header + at least two entries
    for row in rows[1:]:
        duration = int(row[7])
        assert duration >= 1
    shutdown_logger()
