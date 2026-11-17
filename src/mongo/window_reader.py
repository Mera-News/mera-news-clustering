"""Rolling-window read of `news-article` embeddings from Mongo.

Window semantics: read every article created in the last
`CLUSTER_WINDOW_HOURS` that carries a non-empty text-matching embedding vector,
project only what clustering + persistence need, and hand the caller a plain
list of docs whose vector is a float32 ndarray, not the driver's list.

Memory: pymongo decodes a BSON double array into a Python list of float objects,
measured at ~37 bytes per element (list pointer + float object) against 4 for
float32. At a 550k x 768 window that is ~15-16 GB of host RAM, so each vector is
converted to float32 as the cursor streams and the list is dropped at once; only
one cursor batch of lists is ever alive.

Dedup, deliberate:
  * `article_url` is projected and the window is deduped by it (highest
    `createdAt` wins): an at-least-once / re-ingested duplicate would
    otherwise seed a fake 2-point cluster at `min_cluster_size=2`.

Pure compute + a pymongo read — no HDBSCAN, no HTTP. The `db` handle is
injected by the caller.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

# The text-matching embedding field clustering fits on (metadata-free vector —
# "match by content, not geography"), as mera-news-data-server writes it. The
# `_internal_only`-suffixed name is a different, unused field.
WINDOW_VECTOR_FIELD = "embedding_article_data_text_matching"

# Projection: identity (_id) + dedup key (article_url) + tie-breaker (createdAt)
# + the vector. Nothing else — persistence only needs _id, and stamping keys on
# _id too.
WINDOW_PROJECTION = {
    "_id": 1,
    "article_url": 1,
    "createdAt": 1,
    WINDOW_VECTOR_FIELD: 1,
}


def build_window_query(cutoff: datetime) -> dict:
    """The `news-article` filter for the rolling window: created at-or-after the
    cutoff AND carrying a present, non-empty text-matching vector (an
    `$exists` + `$ne: []` predicate)."""
    return {
        "createdAt": {"$gte": cutoff},
        WINDOW_VECTOR_FIELD: {"$exists": True, "$ne": []},
    }


def build_window_query_range(lo: datetime, hi: datetime, *, inclusive_hi: bool) -> dict:
    """Same window predicate as `build_window_query`, bounded to a createdAt
    sub-range `[lo, hi)` (or `[lo, hi]` when `inclusive_hi`). Used by the
    parallel segmented fetch so N threads read disjoint createdAt slices."""
    created = {"$gte": lo, ("$lte" if inclusive_hi else "$lt"): hi}
    return {
        "createdAt": created,
        WINDOW_VECTOR_FIELD: {"$exists": True, "$ne": []},
    }


def _split_time_ranges(cutoff: datetime, now: datetime, n: int) -> list[tuple[datetime, datetime, bool]]:
    """Split `[cutoff, now]` into `n` contiguous createdAt sub-ranges.

    Returns `(lo, hi, inclusive_hi)` triples with no gaps/overlaps: each range's
    `hi` equals the next range's `lo`, all but the last are half-open (`$lt hi`),
    and the last is inclusive (`$lte now`) so the top boundary matches the
    sequential path's behavior.
    """
    span = (now - cutoff) / n
    ranges: list[tuple[datetime, datetime, bool]] = []
    for i in range(n):
        lo = cutoff + span * i
        last = i == n - 1
        hi = now if last else cutoff + span * (i + 1)
        ranges.append((lo, hi, last))
    return ranges


def _created(doc: dict) -> datetime:
    ts = doc.get("createdAt")
    if isinstance(ts, datetime):
        return ts
    # Missing/odd createdAt sorts oldest so a doc that DOES have one wins the
    # dedup — never crash the pass on a malformed timestamp.
    return datetime.min.replace(tzinfo=timezone.utc)


def dedup_by_url(docs: list[dict]) -> list[dict]:
    """Collapse duplicate `article_url`s, keeping the newest (`createdAt`) copy.

    Docs with no `article_url` can't collide, so they're keyed by `_id` and all
    kept. Returns docs in ascending (winning) `createdAt`, then `_id`, order for
    a deterministic downstream fit.
    """
    best: dict = {}
    for doc in docs:
        url = doc.get("article_url")
        key = url if url else ("__id__", doc.get("_id"))
        current = best.get(key)
        if current is None or _created(doc) > _created(current):
            best[key] = doc
    return sorted(best.values(), key=lambda d: (_created(d), str(d.get("_id"))))


class MongoWindowReader:
    """Reads the rolling embedding window from `news-article`. Config supplies
    the window length (`CLUSTER_WINDOW_HOURS`); `read(db, now)` returns the
    deduped list of window docs (each with `_id`, `article_url`, and the
    `WINDOW_VECTOR_FIELD` vector as a float32 ndarray)."""

    def __init__(self, config):
        self._config = config
        self._window_hours = config.CLUSTER_WINDOW_HOURS
        self._batch_size = getattr(config, "FETCH_BATCH_SIZE", 0)
        self._parallelism = getattr(config, "FETCH_PARALLELISM", 1)

    def read(self, db, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self._window_hours)
        if self._parallelism > 1:
            docs = self._read_parallel(db, cutoff, now)
        else:
            docs = self._read_query(db, build_window_query(cutoff))
        return dedup_by_url(docs)

    def _read_query(self, db, query: dict) -> list[dict]:
        cursor = db["news-article"].find(query, WINDOW_PROJECTION)
        # Larger cursor batch size = fewer getMore round-trips over the wire.
        if self._batch_size and hasattr(cursor, "batch_size"):
            cursor = cursor.batch_size(self._batch_size)
        # Defensive re-check of non-emptiness: the $ne:[] filter excludes empty
        # arrays, but a missing-key edge or a null slips through some drivers.
        docs = []
        for d in cursor:
            vec = d.get(WINDOW_VECTOR_FIELD)
            if vec is not None and len(vec):
                d[WINDOW_VECTOR_FIELD] = np.asarray(vec, dtype=np.float32)
                docs.append(d)
        return docs

    def _read_parallel(self, db, cutoff: datetime, now: datetime) -> list[dict]:
        ranges = _split_time_ranges(cutoff, now, self._parallelism)

        def _fetch(rng):
            lo, hi, inclusive = rng
            return self._read_query(db, build_window_query_range(lo, hi, inclusive_hi=inclusive))

        with ThreadPoolExecutor(max_workers=self._parallelism) as pool:
            results = [f.result() for f in [pool.submit(_fetch, rng) for rng in ranges]]
        docs: list[dict] = []
        for chunk in results:
            docs.extend(chunk)
        return docs
