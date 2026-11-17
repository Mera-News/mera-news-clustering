"""Structured JSON logger for the clustering job. Same line shape as the
mera-news-data-server JobLogger, so log-based metric filters read both
identically.

- One JSON object per line to stdout, GCP-severity tagged.
- Levels are the alerting contract, not style:
    info  — cumulative run narrative (`run_start`, `chunk_done`, `run_summary`)
    warn  — recoverable anomalies (`schema_invalid_dropped`, dropped items)
    error — infra-fatal only (accompanies the non-zero exit)
    debug — RESERVED. No debug lines are written by this codebase.
- Default level: info.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Callable

_LEVEL_ORDER = {"debug": 10, "info": 20, "warn": 30, "error": 40}
_GCP_SEVERITY = {"debug": "DEBUG", "info": "INFO", "warn": "WARNING", "error": "ERROR"}


def parse_level(raw: str | None) -> str:
    """Parse a LOG_LEVEL string, falling back to `info` for anything unknown."""
    v = (raw or "").strip().lower()
    return v if v in _LEVEL_ORDER else "info"


class JobLogger:
    def __init__(
        self,
        job: str,
        level: str = "info",
        write: Callable[[str], None] | None = None,
        now: Callable[[], datetime] | None = None,
    ):
        self.job = job
        self._threshold = _LEVEL_ORDER[parse_level(level)]
        self._write = write or (lambda line: sys.stdout.write(line + "\n"))
        self._now = now or (lambda: datetime.now(timezone.utc))

    def _emit(self, level: str, message: str, fields: dict[str, Any]) -> None:
        if _LEVEL_ORDER[level] < self._threshold:
            return
        entry = {
            "severity": _GCP_SEVERITY[level],
            "level": level,
            "timestamp": self._now().isoformat(),
            "job": self.job,
            "message": message,
            **fields,
        }
        self._write(json.dumps(entry, default=str))

    def info(self, message: str, **fields: Any) -> None:
        self._emit("info", message, fields)

    def warn(self, message: str, **fields: Any) -> None:
        self._emit("warn", message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit("error", message, fields)

    def debug(self, message: str, **fields: Any) -> None:
        # Reserved. Intentionally emits nothing unless LOG_LEVEL=debug. Do NOT
        # add debug logging in this codebase.
        self._emit("debug", message, fields)
