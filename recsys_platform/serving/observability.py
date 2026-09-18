"""Lightweight observability primitives for model serving."""

import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock
from time import perf_counter_ns
from typing import Any


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """Aggregated measurements for one serving operation."""

    operation: str
    labels: dict[str, str]
    count: int
    failures: int
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass(slots=True)
class _MetricStats:
    count: int = 0
    failures: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, duration_ms: float, *, success: bool) -> None:
        self.count += 1
        self.failures += int(not success)
        self.total_ms += duration_ms
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


type MetricKey = tuple[str, tuple[tuple[str, str], ...]]


class Observer:
    """Measure, aggregate, and log serving operations.

    Measurements are aggregated by operation name and low-cardinality labels.
    Every completed operation is also emitted as a structured log record.

    The observer is thread-safe and intended to be shared by the serving
    application, model manager, and inference routes.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("recsys_platform.serving")

        self._metrics: dict[MetricKey, _MetricStats] = {}
        self._lock = Lock()

    @contextmanager
    def measure(
        self, operation: str, *, labels: Mapping[str, str] | None = None, **fields: Any
    ) -> Iterator[None]:
        """Measure one operation and record its outcome.

        Args:
            operation: Stable operation name, such as ``"inference"`` or
                ``"model_load"``.
            labels: Low-cardinality dimensions used to group measurements.
            **fields: Additional values included in the log event but not
                used for metric aggregation.
        """
        labels = dict(labels or {})
        start = perf_counter_ns()
        success = False
        error_type: str | None = None

        try:
            yield
            success = True
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            duration_ms = (perf_counter_ns() - start) / 1_000_000

            self._record(
                operation=operation, labels=labels, duration_ms=duration_ms, success=success
            )

            self._log(
                operation=operation,
                labels=labels,
                duration_ms=duration_ms,
                success=success,
                error_type=error_type,
                fields=fields,
            )

    def snapshot(self) -> tuple[MetricSnapshot, ...]:
        """Return a consistent snapshot of aggregated measurements."""
        with self._lock:
            items = tuple(self._metrics.items())

        return tuple(
            MetricSnapshot(
                operation=operation,
                labels=dict(labels),
                count=stats.count,
                failures=stats.failures,
                mean_ms=stats.total_ms / stats.count,
                min_ms=stats.min_ms,
                max_ms=stats.max_ms,
            )
            for (operation, labels), stats in items
        )

    def _record(
        self, *, operation: str, labels: Mapping[str, str], duration_ms: float, success: bool
    ) -> None:
        key = (operation, tuple(sorted(labels.items())))

        with self._lock:
            stats = self._metrics.setdefault(key, _MetricStats())

            stats.record(duration_ms, success=success)

    def _log(
        self,
        *,
        operation: str,
        labels: Mapping[str, str],
        duration_ms: float,
        success: bool,
        error_type: str | None,
        fields: Mapping[str, Any],
    ) -> None:
        event = {
            "operation": operation,
            "duration_ms": round(duration_ms, 3),
            "success": success,
            **labels,
            **fields,
        }

        if error_type is not None:
            event["error_type"] = error_type

        message = json.dumps(event, separators=(",", ":"), default=str)

        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)
