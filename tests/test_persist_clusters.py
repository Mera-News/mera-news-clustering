"""Unit tests for src/mongo/persist_clusters.py pure helpers. The persist itself
is covered by the integration tests against a real mongod."""
from __future__ import annotations

from src.mongo.persist_clusters import _now


def test_now_is_tz_aware_utc():
    assert _now().tzinfo is not None
