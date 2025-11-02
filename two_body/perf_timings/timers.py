from __future__ import annotations

import contextlib
import functools
import time
from typing import Any, Callable, Optional

from .logger import PERF_TIMINGS_ENABLED, TimingLogger, get_timing_logger


def _resolve_value(value: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if callable(value):
        try:
            return value(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return {"resolver_error": str(exc)}
    return value


def _to_int(value: Any, default: int = -1) -> int:
    if callable(value):
        try:
            value = value()
        except Exception:
            return default
    try:
        return int(value)
    except Exception:
        return default


class _TimingBlock(contextlib.ContextDecorator):
    def __init__(
        self,
        section: str,
        *,
        epoch: Any = -1,
        batch_id: Any = -1,
        individual_id: Any = -1,
        extra: Optional[Any] = None,
    ) -> None:
        self.section = section
        self.epoch = epoch
        self.batch_id = batch_id
        self.individual_id = individual_id
        self.extra = extra
        self._start_ns: Optional[int] = None
        self._logger: Optional[TimingLogger] = None

    def __enter__(self) -> "_TimingBlock":
        if not PERF_TIMINGS_ENABLED:
            return self
        self._start_ns = time.perf_counter_ns()
        try:
            self._logger = get_timing_logger()
        except Exception:
            self._logger = None
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if not PERF_TIMINGS_ENABLED or self._logger is None or self._start_ns is None:
            return False
        end_ns = time.perf_counter_ns()
        payload = None
        try:
            payload = self.extra() if callable(self.extra) else self.extra
        except Exception as exc:  # noqa: BLE001
            payload = {"extra_error": str(exc)}
        try:
            self._logger.record(
                section=self.section,
                start_ns=self._start_ns,
                end_ns=end_ns,
                epoch=_to_int(self.epoch),
                batch_id=_to_int(self.batch_id),
                individual_id=_to_int(self.individual_id),
                extra=payload,
            )
        except Exception:
            # Errors should never interrupt the host code.
            pass
        return False


def time_block(
    section: str,
    *,
    epoch: Any = -1,
    batch_id: Any = -1,
    individual_id: Any = -1,
    extra: Optional[Any] = None,
) -> _TimingBlock:
    """
    Context manager for ad-hoc timing blocks.
    """

    return _TimingBlock(
        section,
        epoch=epoch,
        batch_id=batch_id,
        individual_id=individual_id,
        extra=extra,
    )


def time_section(
    section: str,
    *,
    epoch: Any = -1,
    batch_id: Any = -1,
    individual_id: Any = -1,
    extra: Optional[Any] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator that records execution time for the wrapped callable.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PERF_TIMINGS_ENABLED:
                return func(*args, **kwargs)

            resolved_epoch = _resolve_value(epoch, args, kwargs)
            resolved_batch = _resolve_value(batch_id, args, kwargs)
            resolved_individual = _resolve_value(individual_id, args, kwargs)
            resolved_extra = _resolve_value(extra, args, kwargs)

            with time_block(
                section,
                epoch=resolved_epoch,
                batch_id=resolved_batch,
                individual_id=resolved_individual,
                extra=resolved_extra,
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator
