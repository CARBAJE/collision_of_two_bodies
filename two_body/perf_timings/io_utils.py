from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from .logger import CSV_HEADER, DEFAULT_TIMINGS_DIR

TIMINGS_DIR = DEFAULT_TIMINGS_DIR
_PROJECT_ROOT = TIMINGS_DIR.parent.parent
REPORTS_DIR = _PROJECT_ROOT / "reports"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_timing_csvs(base_dir: Optional[Path | str] = None) -> List[Path]:
    base = Path(base_dir) if base_dir else TIMINGS_DIR
    if not base.exists():
        return []
    return sorted(base.glob("timings_*.csv"), key=lambda p: p.stat().st_mtime)


def latest_timing_csv(base_dir: Optional[Path | str] = None) -> Optional[Path]:
    files = list_timing_csvs(base_dir)
    return files[-1] if files else None


def read_timings_csv(path: Path) -> List[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if header != CSV_HEADER:
            raise ValueError(
                f"Archivo {path} tiene cabecera inesperada {header}; se esperaba {CSV_HEADER}."
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(_cast_row(row))
    return rows


def filter_rows(
    rows: Sequence[dict[str, Any]],
    *,
    run_id: Optional[str] = None,
    epoch: Optional[int] = None,
    batch_id: Optional[int] = None,
    sections: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    if sections is not None:
        wanted = {sec.strip() for sec in sections if sec.strip()}
    else:
        wanted = None
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if run_id is not None and row.get("run_id") != run_id:
            continue
        if epoch is not None and row.get("epoch") != epoch:
            continue
        if batch_id is not None and row.get("batch_id") != batch_id:
            continue
        if wanted is not None and row.get("section") not in wanted:
            continue
        filtered.append(row)
    return filtered


def parse_sections_arg(raw: Optional[str]) -> Optional[list[str]]:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",")]
    return [p for p in parts if p]


def _cast_row(raw: dict[str, Any]) -> dict[str, Any]:
    def _to_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except Exception:
            return default

    start_ns = _to_int(raw.get("start_ns"), 0)
    end_ns = _to_int(raw.get("end_ns"), start_ns)
    computed_duration = max(1, (end_ns - start_ns) // 1_000)
    duration_us = _to_int(raw.get("duration_us"), computed_duration)
    extra_raw = raw.get("extra", "")
    extra = {}
    if isinstance(extra_raw, str) and extra_raw.strip():
        try:
            extra = json.loads(extra_raw)
        except Exception:
            extra = {"raw": extra_raw}

    return {
        "run_id": str(raw.get("run_id", "")),
        "epoch": _to_int(raw.get("epoch")),
        "batch_id": _to_int(raw.get("batch_id")),
        "individual_id": _to_int(raw.get("individual_id")),
        "section": str(raw.get("section", "")),
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_us": duration_us,
        "extra": extra,
    }
