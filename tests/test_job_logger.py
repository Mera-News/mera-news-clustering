"""Unit tests for src/job_logger.py — structured JSON logger."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src.job_logger import JobLogger, parse_level


class TestParseLevel:
    def test_known_levels(self):
        for level in ("debug", "info", "warn", "error"):
            assert parse_level(level) == level

    def test_case_insensitive(self):
        assert parse_level("INFO") == "info"

    def test_unknown_falls_back_to_info(self):
        assert parse_level("verbose") == "info"

    def test_none_falls_back_to_info(self):
        assert parse_level(None) == "info"


def _logger(level="info"):
    lines: list[dict] = []
    log = JobLogger(
        "news-clustering-job",
        level=level,
        write=lambda ln: lines.append(json.loads(ln)),
        now=lambda: datetime(2026, 7, 6, tzinfo=timezone.utc),
    )
    return log, lines


class TestEmission:
    def test_info_line_shape(self):
        log, lines = _logger()
        log.info("run_start", pending=5, endOffsets={0: 10})
        entry = lines[0]
        assert entry["severity"] == "INFO"
        assert entry["level"] == "info"
        assert entry["job"] == "news-clustering-job"
        assert entry["message"] == "run_start"
        assert entry["pending"] == 5
        assert entry["endOffsets"] == {"0": 10}  # json coerces int keys to str
        assert entry["timestamp"] == "2026-07-06T00:00:00+00:00"

    def test_warn_severity(self):
        log, lines = _logger()
        log.warn("schema_invalid_dropped", id="k1", reason="bad")
        assert lines[0]["severity"] == "WARNING"

    def test_error_severity(self):
        log, lines = _logger()
        log.error("infra_fatal", error="broker down")
        assert lines[0]["severity"] == "ERROR"


class TestLevelFiltering:
    def test_info_suppressed_when_threshold_warn(self):
        log, lines = _logger(level="warn")
        log.info("run_start")
        assert lines == []

    def test_warn_emitted_when_threshold_warn(self):
        log, lines = _logger(level="warn")
        log.warn("dropped")
        assert len(lines) == 1

    def test_debug_never_emitted_at_info(self):
        log, lines = _logger(level="info")
        log.debug("detail")
        assert lines == []

    def test_debug_emitted_only_at_debug_level(self):
        log, lines = _logger(level="debug")
        log.debug("detail")
        assert len(lines) == 1
        assert lines[0]["severity"] == "DEBUG"
