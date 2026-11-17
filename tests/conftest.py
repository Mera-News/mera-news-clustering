"""Shared pytest fixtures: settings and a structured-log sink."""
from __future__ import annotations

import pytest

from src.config import Settings


@pytest.fixture
def make_settings():
    def _make(**overrides) -> Settings:
        return Settings(**overrides)

    return _make


@pytest.fixture
def settings(make_settings) -> Settings:
    return make_settings()


# --------------------------------------------------------------------------- #
# Structured-log capture: a JobLogger sink that records parsed JSON lines.
# --------------------------------------------------------------------------- #
@pytest.fixture
def log_sink():
    import json

    lines: list[dict] = []

    def write(line: str) -> None:
        lines.append(json.loads(line))

    write.lines = lines  # type: ignore[attr-defined]
    return write
