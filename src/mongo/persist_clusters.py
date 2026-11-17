"""Persist the `news-cluster` + `cluster-article-link` snapshot (generation swap).

Doc shapes and field names are fixed: readers in mera-news-data-server depend
on them (no field or collection renames):

  news-cluster        : { _id, clusterSize, createdAt, updatedAt }
                        (+ clusteringRunId; + stableClusterId on multi-member)
  cluster-article-link: { clusterId, articleId, confidence, createdAt }
                        (+ clusteringRunId — additive, field+index already exist
                         in the server's Mongoose schema)
  news-article stamp  : processingStatus="aggregation_successful",
                        aggregationCompletedAt, updatedAt

How this differs from the original wipe-and-insert:

  * Generation swap (default, PERSIST_LEGACY_WIPE_FIRST=False): the new
    generation is INSERTED FIRST (cluster docs, then link docs, each stamped with
    this run's `clusteringRunId`), and only THEN are prior generations removed via
    `delete_many({"clusteringRunId": {"$ne": clustering_run_id}})` on each
    collection. `$ne` matches docs that are missing the field, so pre-deploy rows
    are cleaned on the first run. Consumers therefore never observe an empty
    collection mid-run (the old wipe-first order left both collections empty for
    the whole insert). A crash BETWEEN the inserts and the deletes leaves two
    generations behind until the next run's delete sweeps the stale one — the
    greedy membership-overlap stable-id matcher tolerates that (accepted edge).

  * Legacy wipe-first (PERSIST_LEGACY_WIPE_FIRST=True): preserves the original
    order exactly — delete_many({}) on both collections first, then insert, then
    stamp.

  * Singleton skip (PERSIST_SINGLETON_CLUSTERS=False): size-1 clusters get no
    cluster doc and no link docs. ALL window article ids are still stamped
    regardless — the stamp is the article-side signal and must not depend on
    whether the singleton was materialized.

  * Inserts are unpaced + parallel (thread pool, see _insert_many_parallel) —
    the original 256-doc chunks + 0.15s inter-chunk sleep are gone. A non-zero
    DB_WRITE_PAUSE_S forces the sequential paced fallback (emergency throttle).

  * The article stamp is chunked AND filtered: each chunk excludes docs already
    `processingStatus="aggregation_successful"` so re-stamping the (mostly
    unchanged) window is not re-written every run.
"""
from __future__ import annotations

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from bson import ObjectId

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB write helpers.
# ---------------------------------------------------------------------------

def _bulk_write_paced(collection, ops: list, *, batch_size: int, pause: float) -> None:
    """Sequential paced bulk_write (retained from the original job for parity/tests)."""
    for start in range(0, len(ops), batch_size):
        chunk = ops[start : start + batch_size]
        collection.bulk_write(chunk, ordered=False)
        if start + batch_size < len(ops):
            time.sleep(pause)


def _insert_many_paced(collection, docs: list, *, batch_size: int, pause: float) -> None:
    """Sequential paced insert_many (emergency-throttle fallback for pause > 0)."""
    for start in range(0, len(docs), batch_size):
        chunk = docs[start : start + batch_size]
        collection.insert_many(chunk, ordered=False)
        if start + batch_size < len(docs):
            time.sleep(pause)


def _insert_many_parallel(
    collection, docs: list, *, batch_size: int, concurrency: int, pause: float = 0.0
) -> None:
    """Chunk `docs` at `batch_size` and insert the chunks in parallel.

    If `pause > 0`, falls back to the sequential paced loop (emergency throttle).
    Otherwise submits each chunk's `insert_many(chunk, ordered=False)` to a
    `ThreadPoolExecutor(max_workers=max(1, concurrency))` and re-raises the first
    worker exception (persist failure is infra-fatal upstream).
    """
    if not docs:
        return
    if pause > 0:
        _insert_many_paced(collection, docs, batch_size=batch_size, pause=pause)
        return
    chunks = [docs[start : start + batch_size] for start in range(0, len(docs), batch_size)]
    workers = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collection.insert_many, chunk, False) for chunk in chunks]
        for future in futures:
            future.result()  # propagate worker exceptions


def _stamp_articles_chunked(
    col_articles, ids: list, set_fields: dict, *, batch_size: int, concurrency: int
) -> tuple[int, int]:
    """Stamp `set_fields` onto the window article ids, chunked and filtered.

    Each chunk runs `update_many({"_id": {"$in": chunk}, "processingStatus":
    {"$ne": "aggregation_successful"}}, {"$set": set_fields})` — the filter skips
    docs already stamped so the (mostly unchanged) window is not re-written every
    run. Chunks are fanned across the thread pool. Returns aggregate
    (matched, modified) counts.
    """
    if not ids:
        return 0, 0
    chunks = [ids[start : start + batch_size] for start in range(0, len(ids), batch_size)]

    def _run(chunk):
        return col_articles.update_many(
            {"_id": {"$in": chunk}, "processingStatus": {"$ne": "aggregation_successful"}},
            {"$set": set_fields},
        )

    workers = max(1, concurrency)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, chunk) for chunk in chunks]
        results = [future.result() for future in futures]

    matched = sum(r.matched_count for r in results)
    modified = sum(r.modified_count for r in results)
    return matched, modified


def _now() -> datetime:
    return datetime.now(timezone.utc)


def read_previous_generation(db) -> tuple[dict[str, str], dict[str, str]]:
    """Load the previous run's stable-cluster identity so this run can carry
    `stableClusterId`s across the persist.

    MUST be called strictly BEFORE `persist_clusters` inserts/deletes. Under the
    generation swap the new docs land BEFORE the old generation is deleted, so if
    this read ran after the insert it would see both generations and the overlap
    matcher would compare a generation against itself. A crash between the inserts
    and the deletes leaves two generations until the next run's delete sweeps the
    stale one; the greedy overlap matcher tolerates that (accepted edge). Under
    the legacy wipe-first path the collections are emptied first, so it must run
    before that too.

    Reads (projections only, one indexed scan each):
      * `news-cluster` docs with `clusterSize > 1` AND a `stableClusterId` — the
        previous multi-member clusters.
      * `cluster-article-link` rows for those clusters — their memberships.

    Returns:
      prev_membership : { str(articleId) -> str(prevClusterId) }
      prev_stable_ids : { str(prevClusterId) -> stableClusterId }
    """
    col_clusters = db["news-cluster"]
    col_links = db["cluster-article-link"]

    prev_stable_ids: dict[str, str] = {}
    for doc in col_clusters.find(
        {"clusterSize": {"$gt": 1}, "stableClusterId": {"$exists": True}},
        {"_id": 1, "stableClusterId": 1},
    ):
        sid = doc.get("stableClusterId")
        if sid:
            prev_stable_ids[str(doc["_id"])] = sid

    prev_membership: dict[str, str] = {}
    if prev_stable_ids:
        cluster_ids = [ObjectId(cid) for cid in prev_stable_ids]
        for link in col_links.find(
            {"clusterId": {"$in": cluster_ids}},
            {"_id": 0, "clusterId": 1, "articleId": 1},
        ):
            prev_membership[str(link["articleId"])] = str(link["clusterId"])

    return prev_membership, prev_stable_ids


def persist_clusters(
    db,
    clusters: list[dict],
    config,
    *,
    clustering_run_id: str | None = None,
    now: datetime | None = None,
    logger=None,
) -> dict:
    """Write this run's HDBSCAN output to `news-cluster` + `cluster-article-link`
    and stamp the source articles `aggregation_successful`.

    `clusters` is a list of {"members": [(ObjectId, confidence), ...]} groupings;
    multi-member clusters may carry a `stableClusterId` (attached by the caller).

    Order (default generation swap): insert new cluster docs → insert new link
    docs → delete prior generations by `clusteringRunId` on each collection →
    stamp articles. `PERSIST_LEGACY_WIPE_FIRST=True` restores the original
    wipe-first order. `PERSIST_SINGLETON_CLUSTERS=False` skips materializing
    size-1 clusters (articles are stamped regardless).

    When `logger` (a JobLogger) is provided, emits one `persist_substages` info
    event with per-stage timings and counts. Returns
    `{"clusters", "clusterDocs", "linkDocs"}` — `clusters` is the total number of
    input groupings; `clusterDocs`/`linkDocs` are the docs actually written.
    """
    if not clusters:
        return {"clusters": 0, "clusterDocs": 0, "linkDocs": 0}

    clustering_run_id = clustering_run_id or str(uuid.uuid4())
    now = now or _now()

    col_clusters = db["news-cluster"]
    col_links = db["cluster-article-link"]
    col_articles = db["news-article"]
    bsz = config.DB_WRITE_BATCH_SIZE
    pause = config.DB_WRITE_PAUSE_S
    concurrency = max(1, min(config.DB_WRITE_CONCURRENCY, config.MONGODB_MAX_POOL_SIZE))
    keep_singletons = config.PERSIST_SINGLETON_CLUSTERS
    legacy_wipe = config.PERSIST_LEGACY_WIPE_FIRST

    # ALL window article ids are stamped, independent of the singleton skip.
    all_article_ids: list[ObjectId] = [
        article_id for cluster in clusters for article_id, _ in cluster["members"]
    ]

    new_cluster_docs: list = []
    link_docs: list = []
    for cluster in clusters:
        members: list[tuple[ObjectId, float]] = cluster["members"]
        if not keep_singletons and len(members) == 1:
            continue
        cluster_id = ObjectId()
        cluster_doc = {
            "_id": cluster_id,
            # Total members across the rolling window (denormalized so readers
            # don't count link rows). Singletons/noise are size 1.
            "clusterSize": len(members),
            # Trace which run minted this cluster — and the generation-swap key.
            "clusteringRunId": clustering_run_id,
            "createdAt": now,
            "updatedAt": now,
        }
        stable_cluster_id = cluster.get("stableClusterId")
        if stable_cluster_id is not None:
            cluster_doc["stableClusterId"] = stable_cluster_id
        new_cluster_docs.append(cluster_doc)
        for article_id, confidence in members:
            link_docs.append({
                "clusterId": cluster_id,
                "articleId": article_id,
                "confidence": float(confidence),
                "createdAt": now,
                # Additive generation-swap key (field + index exist server-side).
                "clusteringRunId": clustering_run_id,
            })

    cluster_insert_ms = link_insert_ms = delete_old_gen_ms = stamp_ms = 0
    deleted_links = deleted_clusters = 0

    if legacy_wipe:
        # Original order: wipe both collections first, then insert.
        t0 = time.monotonic()
        deleted_links = col_links.delete_many({}).deleted_count
        deleted_clusters = col_clusters.delete_many({}).deleted_count
        delete_old_gen_ms = int((time.monotonic() - t0) * 1000)

        t0 = time.monotonic()
        _insert_many_parallel(col_clusters, new_cluster_docs, batch_size=bsz, concurrency=concurrency, pause=pause)
        cluster_insert_ms = int((time.monotonic() - t0) * 1000)
        t0 = time.monotonic()
        _insert_many_parallel(col_links, link_docs, batch_size=bsz, concurrency=concurrency, pause=pause)
        link_insert_ms = int((time.monotonic() - t0) * 1000)
    else:
        # Generation swap: insert the new generation FIRST, then drop prior ones.
        t0 = time.monotonic()
        _insert_many_parallel(col_clusters, new_cluster_docs, batch_size=bsz, concurrency=concurrency, pause=pause)
        cluster_insert_ms = int((time.monotonic() - t0) * 1000)
        t0 = time.monotonic()
        _insert_many_parallel(col_links, link_docs, batch_size=bsz, concurrency=concurrency, pause=pause)
        link_insert_ms = int((time.monotonic() - t0) * 1000)

        t0 = time.monotonic()
        deleted_links = col_links.delete_many(
            {"clusteringRunId": {"$ne": clustering_run_id}}
        ).deleted_count
        deleted_clusters = col_clusters.delete_many(
            {"clusteringRunId": {"$ne": clustering_run_id}}
        ).deleted_count
        delete_old_gen_ms = int((time.monotonic() - t0) * 1000)

    stamped_matched = stamped_modified = 0
    if all_article_ids:
        t0 = time.monotonic()
        stamped_matched, stamped_modified = _stamp_articles_chunked(
            col_articles,
            all_article_ids,
            {
                "processingStatus": "aggregation_successful",
                "aggregationCompletedAt": now,
                "updatedAt": now,
            },
            batch_size=config.STAMP_BATCH_SIZE,
            concurrency=concurrency,
        )
        stamp_ms = int((time.monotonic() - t0) * 1000)

    if logger is not None:
        logger.info(
            "persist_substages",
            clusterInsertMs=cluster_insert_ms,
            linkInsertMs=link_insert_ms,
            deleteOldGenMs=delete_old_gen_ms,
            stampMs=stamp_ms,
            clusterDocs=len(new_cluster_docs),
            linkDocs=len(link_docs),
            stampedMatched=stamped_matched,
            stampedModified=stamped_modified,
            deletedLinks=deleted_links,
            deletedClusters=deleted_clusters,
        )

    _log.info(
        "Persisted %d clusters (%d cluster docs, %d links); deleted %d stale clusters, %d stale links",
        len(clusters), len(new_cluster_docs), len(link_docs), deleted_clusters, deleted_links,
    )
    return {
        "clusters": len(clusters),
        "clusterDocs": len(new_cluster_docs),
        "linkDocs": len(link_docs),
    }
