from __future__ import annotations

"""
Utilities for high-resolution performance instrumentation across the project.
"""

from .io_utils import (
    REPORTS_DIR,
    TIMINGS_DIR,
    filter_rows,
    latest_timing_csv,
    list_timing_csvs,
    parse_sections_arg,
    read_timings_csv,
)
from .logger import (
    CSV_HEADER,
    PERF_TIMINGS_ENABLED,
    TimingLogger,
    configure_global_logger,
    get_timing_logger,
    shutdown_logger,
)
from .timers import time_block, time_section

__all__ = [
    "CSV_HEADER",
    "PERF_TIMINGS_ENABLED",
    "TimingLogger",
    "configure_global_logger",
    "get_timing_logger",
    "shutdown_logger",
    "time_block",
    "time_section",
    "TIMINGS_DIR",
    "REPORTS_DIR",
    "list_timing_csvs",
    "latest_timing_csv",
    "read_timings_csv",
    "filter_rows",
    "parse_sections_arg",
]
