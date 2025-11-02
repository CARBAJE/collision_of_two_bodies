from __future__ import annotations

import atexit
import csv
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

CSV_HEADER: list[str] = [
    "run_id",
    "epoch",
    "batch_id",
    "individual_id",
    "section",
    "start_ns",
    "end_ns",
    "duration_us",
    "extra",
]


def _env_flag(name: str, default: str = "1") -> bool:
    raw = os.getenv(name, default)
    if raw is None:
        return False
    return raw.strip().lower() not in {"0", "false", "off", "no", ""}


PERF_TIMINGS_ENABLED = _env_flag("PERF_TIMINGS_ENABLED", "1")
PERF_TIMINGS_JSONL = _env_flag("PERF_TIMINGS_JSONL", "0")

DEFAULT_BUFFER_SIZE = 64
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMINGS_DIR = _PACKAGE_ROOT / "data" / "timings"


@dataclass(frozen=True)
class TimingRecord:
    run_id: str
    epoch: int
    batch_id: int
    individual_id: int
    section: str
    start_ns: int
    end_ns: int
    duration_us: int
    extra: str

    def as_row(self) -> list[Any]:
        return [
            self.run_id,
            str(self.epoch),
            str(self.batch_id),
            str(self.individual_id),
            self.section,
            str(self.start_ns),
            str(self.end_ns),
            str(self.duration_us),
            self.extra,
        ]

    def as_jsonl(self) -> str:
        payload = {
            "run_id": self.run_id,
            "epoch": self.epoch,
            "batch_id": self.batch_id,
            "individual_id": self.individual_id,
            "section": self.section,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_us": self.duration_us,
        }
        try:
            payload["extra"] = json.loads(self.extra) if self.extra else {}
        except Exception:
            payload["extra"] = self.extra
        return json.dumps(payload, separators=(",", ":"))


class TimingLogger:
    """
    Buffered CSV/JSONL timing logger.
    """

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        enabled: bool = True,
        base_dir: Optional[Path | str] = None,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        enable_jsonl: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self.enable_jsonl = bool(enable_jsonl)
        try:
            run_uuid = uuid.UUID(run_id) if run_id else uuid.uuid4()
        except Exception:
            run_uuid = uuid.uuid4()
        if getattr(run_uuid, "version", 4) != 4:
            run_uuid = uuid.uuid4()
        self.run_id = str(run_uuid)
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_TIMINGS_DIR
        self.buffer_size = max(1, int(buffer_size))
        self.csv_path = self.base_dir / f"timings_{self.run_id}_{self.timestamp}.csv"
        self.jsonl_path = self.base_dir / f"timings_{self.run_id}_{self.timestamp}.jsonl"
        self._buffer: list[TimingRecord] = []
        self._lock = threading.Lock()
        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.writer] = None
        self._jsonl_file: Optional[Any] = None
        self._closed = False
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    # Public API ---------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.csv_path

    def record(
        self,
        *,
        section: str,
        start_ns: Optional[int],
        end_ns: Optional[int],
        epoch: int = -1,
        batch_id: int = -1,
        individual_id: int = -1,
        extra: Optional[Any] = None,
    ) -> None:
        if not self.enabled:
            return
        try:
            start = int(start_ns if start_ns is not None else time.perf_counter_ns())
            end = int(end_ns if end_ns is not None else time.perf_counter_ns())
            if end < start:
                end = start
            duration_us = max(1, (end - start) // 1_000)
            extra_str = self._serialize_extra(extra)
            record = TimingRecord(
                run_id=self.run_id,
                epoch=int(epoch),
                batch_id=int(batch_id),
                individual_id=int(individual_id),
                section=str(section),
                start_ns=start,
                end_ns=end,
                duration_us=int(duration_us),
                extra=extra_str,
            )
            with self._lock:
                self._buffer.append(record)
                if len(self._buffer) >= self.buffer_size:
                    self._flush_locked()
        except Exception as exc:  # noqa: BLE001
            self._log_error(exc)

    def flush(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            if self._csv_file:
                try:
                    self._csv_file.close()
                except Exception:
                    pass
                finally:
                    self._csv_file = None
            if self._jsonl_file:
                try:
                    self._jsonl_file.close()
                except Exception:
                    pass
                finally:
                    self._jsonl_file = None
            self._closed = True

    # Internal helpers ---------------------------------------------------
    def _ensure_files_locked(self) -> None:
        if self._csv_file is None:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            existed = self.csv_path.exists()
            self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            if not existed:
                self._csv_writer.writerow(CSV_HEADER)
        if self.enable_jsonl and self._jsonl_file is None:
            self._jsonl_file = self.jsonl_path.open("a", encoding="utf-8")

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        self._ensure_files_locked()
        pending = self._buffer
        self._buffer = []
        if self._csv_writer is None or self._csv_file is None:
            return
        try:
            for record in pending:
                self._csv_writer.writerow(record.as_row())
                if self.enable_jsonl and self._jsonl_file is not None:
                    self._jsonl_file.write(record.as_jsonl() + "\n")
            self._csv_file.flush()
            if self.enable_jsonl and self._jsonl_file is not None:
                self._jsonl_file.flush()
        except Exception as exc:  # noqa: BLE001
            self._log_error(exc)

    @staticmethod
    def _serialize_extra(extra: Optional[Any]) -> str:
        if extra is None:
            return "{}"
        if isinstance(extra, str):
            stripped = extra.strip()
            return stripped if stripped else "{}"
        try:
            return json.dumps(extra, separators=(",", ":"), default=str)
        except Exception:
            return json.dumps({"value": str(extra)}, separators=(",", ":"))

    @staticmethod
    def _log_error(exc: Exception) -> None:
        try:
            sys.stderr.write(f"[perf_timings] logger error: {exc}\n")
        except Exception:
            pass


_GLOBAL_LOGGER: Optional[TimingLogger] = None
_GLOBAL_LOCK = threading.Lock()


def get_timing_logger() -> TimingLogger:
    global _GLOBAL_LOGGER
    with _GLOBAL_LOCK:
        if _GLOBAL_LOGGER is None:
            _GLOBAL_LOGGER = TimingLogger(
                enabled=PERF_TIMINGS_ENABLED,
                enable_jsonl=PERF_TIMINGS_JSONL,
            )
        return _GLOBAL_LOGGER


def configure_global_logger(**kwargs: Any) -> TimingLogger:
    global _GLOBAL_LOGGER
    with _GLOBAL_LOCK:
        if _GLOBAL_LOGGER is not None:
            try:
                _GLOBAL_LOGGER.close()
            except Exception:
                pass
        _GLOBAL_LOGGER = TimingLogger(**kwargs)
        return _GLOBAL_LOGGER


def shutdown_logger() -> None:
    global _GLOBAL_LOGGER
    with _GLOBAL_LOCK:
        if _GLOBAL_LOGGER is not None:
            try:
                _GLOBAL_LOGGER.close()
            except Exception:
                pass
            _GLOBAL_LOGGER = None
