"""Unit tests for src/mongo/window_reader.py: query construction, dedup, range split."""
from __future__ import annotations

from datetime import datetime, timezone

from bson import ObjectId

from src.mongo.window_reader import (
    WINDOW_PROJECTION,
    WINDOW_VECTOR_FIELD,
    _split_time_ranges,
    build_window_query,
    build_window_query_range,
    dedup_by_url,
)

UTC = timezone.utc


def _doc(url, created, *, vec=None, _id=None):
    return {
        "_id": _id or ObjectId(),
        "article_url": url,
        "createdAt": created,
        WINDOW_VECTOR_FIELD: vec if vec is not None else [0.1, 0.2],
    }


# --------------------------------------------------------------------------- #
# build_window_query
# --------------------------------------------------------------------------- #
class TestBuildWindowQuery:
    def test_filters_on_createdAt_and_vector(self):
        cutoff = datetime(2026, 7, 1, tzinfo=UTC)
        q = build_window_query(cutoff)
        assert q["createdAt"] == {"$gte": cutoff}
        assert q[WINDOW_VECTOR_FIELD] == {"$exists": True, "$ne": []}

    def test_vector_field_name(self):
        # The field mera-news-data-server writes (not the `_internal_only` name).
        assert WINDOW_VECTOR_FIELD == "embedding_article_data_text_matching"

    def test_projection_limited_to_needed_fields(self):
        assert set(WINDOW_PROJECTION) == {"_id", "article_url", "createdAt", WINDOW_VECTOR_FIELD}


# --------------------------------------------------------------------------- #
# dedup_by_url
# --------------------------------------------------------------------------- #
class TestDedup:
    def test_keeps_newest_by_created(self):
        old = _doc("u", datetime(2026, 7, 1, tzinfo=UTC))
        new = _doc("u", datetime(2026, 7, 2, tzinfo=UTC))
        out = dedup_by_url([old, new])
        assert len(out) == 1
        assert out[0]["_id"] == new["_id"]

    def test_newest_wins_regardless_of_input_order(self):
        new = _doc("u", datetime(2026, 7, 2, tzinfo=UTC))
        old = _doc("u", datetime(2026, 7, 1, tzinfo=UTC))
        out = dedup_by_url([new, old])
        assert out[0]["_id"] == new["_id"]

    def test_distinct_urls_all_kept(self):
        a = _doc("a", datetime(2026, 7, 1, tzinfo=UTC))
        b = _doc("b", datetime(2026, 7, 1, tzinfo=UTC))
        assert len(dedup_by_url([a, b])) == 2

    def test_docs_without_url_all_kept(self):
        a = {"_id": ObjectId(), "createdAt": datetime(2026, 7, 1, tzinfo=UTC), WINDOW_VECTOR_FIELD: [0.1]}
        b = {"_id": ObjectId(), "createdAt": datetime(2026, 7, 1, tzinfo=UTC), WINDOW_VECTOR_FIELD: [0.2]}
        assert len(dedup_by_url([a, b])) == 2

    def test_returned_in_ascending_created_order(self):
        a = _doc("a", datetime(2026, 7, 3, tzinfo=UTC))
        b = _doc("b", datetime(2026, 7, 1, tzinfo=UTC))
        c = _doc("c", datetime(2026, 7, 2, tzinfo=UTC))
        out = dedup_by_url([a, b, c])
        assert [d["article_url"] for d in out] == ["b", "c", "a"]

    def test_missing_created_does_not_crash(self):
        a = {"_id": ObjectId(), "article_url": "u", WINDOW_VECTOR_FIELD: [0.1]}
        b = _doc("u", datetime(2026, 7, 1, tzinfo=UTC))
        # The doc WITH a createdAt wins over the one without.
        out = dedup_by_url([a, b])
        assert len(out) == 1
        assert out[0]["_id"] == b["_id"]


# --------------------------------------------------------------------------- #
# Range split + parallel fetch
# --------------------------------------------------------------------------- #
class TestRangeSplit:
    def test_ranges_cover_window_no_gaps_or_overlaps(self):
        cutoff = datetime(2026, 7, 1, tzinfo=UTC)
        now = datetime(2026, 7, 3, tzinfo=UTC)
        ranges = _split_time_ranges(cutoff, now, 4)
        assert len(ranges) == 4
        assert ranges[0][0] == cutoff
        assert ranges[-1][1] == now
        assert ranges[-1][2] is True  # last range inclusive ($lte now)
        for i in range(3):
            assert ranges[i][2] is False  # non-last are half-open ($lt hi)
            assert ranges[i][1] == ranges[i + 1][0]  # no gap / no overlap

    def test_query_range_bounds(self):
        lo = datetime(2026, 7, 1, tzinfo=UTC)
        hi = datetime(2026, 7, 2, tzinfo=UTC)
        half_open = build_window_query_range(lo, hi, inclusive_hi=False)
        assert half_open["createdAt"] == {"$gte": lo, "$lt": hi}
        closed = build_window_query_range(lo, hi, inclusive_hi=True)
        assert closed["createdAt"] == {"$gte": lo, "$lte": hi}
        assert half_open[WINDOW_VECTOR_FIELD] == {"$exists": True, "$ne": []}
