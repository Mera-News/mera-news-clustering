"""news-clustering-job — Mongo-in → HDBSCAN → Mongo-out.

Mongo IO wrapped around a pure HDBSCAN compute. It is the default
`JOB_ENTRYPOINT` (overridable via env).

Flow (collaborators injected):
  1. singleton guard   — one execution per Cloud Run job.
  2. window read        — 48h of `news-article` embeddings from Mongo.
  3. HDBSCAN fit        — ClusterService.cluster() on the text-matching vectors.
  4. uniquify + size    — noise -1 → unique singletons; per-label sizes.
  5. persist            — insert the new generation then delete the prior one
                          (generation swap), stamp articles `aggregation_successful`.

Clustering is pure Mongo↔Mongo — there is NO callback to the data server.
Finalization (`link-all-topics-to-clusters`) is enqueued by a separate scheduler
cron, not by this job. Only infra failures (Mongo unreachable,
hdbscan/GPU init) exit non-zero; an empty window is a clean exit 0.

Launch:
  prod (L4 GPU): python -m cuml.accel -m src.job_main_mongo
  CPU / CI/dev:  python -m src.job_main_mongo
"""
from __future__ import annotations

CURRENT_JOB = "news-clustering-job"

import os
import resource
import sys
import time
import uuid
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
from bson import ObjectId

from src.config import Settings
from src.job_logger import JobLogger
from src.mongo.persist_clusters import persist_clusters, read_previous_generation
from src.mongo.window_reader import WINDOW_VECTOR_FIELD, MongoWindowReader
from src.services.cluster_service import (
    ClusterService,
    cluster_sizes,
    uniquify_noise_labels,
)
from src.services.stable_id_service import match_stable_ids
from src.singleton import singleton_decision

# Captured at import time so main() can log the import→main gap (dominated by the
# cuml.accel / CUDA import under the GPU launch); combined with the container
# start timestamp this makes that boot cost visible.
_IMPORT_TS = time.monotonic()


# Sleeps between initial Mongo connect attempts. Each attempt can itself spend up
# to serverSelectionTimeoutMS (10s), so 5 attempts + these sleeps ride out a
# ~3 minute network blip before the run gives up as infra_fatal.
MONGO_CONNECT_BACKOFF_S = (10, 20, 40, 60)


class InfraFatal(Exception):
    """Wraps an infra-fatal condition so main() logs + exits non-zero."""


def _mem_fields() -> dict:
    """Host memory at a stage boundary, for the structured log lines.
    peakRssMb = process high-water mark (ru_maxrss: KiB on Linux, bytes on
    macOS); rssMb = current RSS (Linux /proc only); gpuUsedMb = device-wide used
    memory, only when cupy is already loaded (the cuml.accel launch)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fields = {"peakRssMb": peak // 2**20 if sys.platform == "darwin" else peak // 1024}
    try:
        with open("/proc/self/statm") as f:
            fields["rssMb"] = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 2**20
    except (OSError, ValueError, IndexError):
        pass
    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            free, total = cupy.cuda.runtime.memGetInfo()
            fields["gpuUsedMb"] = (total - free) // 2**20
        except Exception:  # noqa: BLE001 - instrumentation must never fail a run
            pass
    return fields


def _connect(client, logger: JobLogger, sleep):
    """Ping Mongo, retrying with backoff (MONGO_CONNECT_BACKOFF_S) so a brief
    outage does not kill the run. The final failure is infra-fatal."""
    for attempt, delay in enumerate((*MONGO_CONNECT_BACKOFF_S, None), start=1):
        try:
            db = client.get_default_database()
            client.admin.command("ping")
            return db
        except Exception as err:  # noqa: BLE001 - Mongo unreachable is infra-fatal
            if delay is None:
                raise InfraFatal(f"mongo connect failed: {err}") from err
            logger.warn("mongo_connect_retry", attempt=attempt, retryInS=delay, error=str(err))
            sleep(delay)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def build_clusters(
    article_ids: list[ObjectId], labels: np.ndarray, probs: np.ndarray
) -> list[dict]:
    """Group window article ids into the persist's cluster shape — a list of
    {"members": [(articleId ObjectId, confidence float), ...]}. `labels` are the
    UNIQUIFIED labels (each negative label is its own singleton noise cluster);
    noise members carry confidence 1.0."""
    groups: dict[int, list[tuple[ObjectId, float]]] = {}
    for i, article_id in enumerate(article_ids):
        label = int(labels[i])
        confidence = float(probs[i]) if label >= 0 else 1.0
        groups.setdefault(label, []).append((article_id, confidence))
    return [{"members": members} for members in groups.values()]


def run_clustering_mongo(
    config: Settings,
    logger: JobLogger,
    client,
    reader: MongoWindowReader,
    cluster_svc: ClusterService,
    *,
    persist=persist_clusters,
    read_prev=read_previous_generation,
    now: datetime | None = None,
    sleep=time.sleep,
) -> dict:
    """Core run. Raises InfraFatal on Mongo/hdbscan/persist failures;
    an empty window returns a clean summary (no clusters). Clustering is pure
    Mongo↔Mongo — finalization is a separate scheduler cron, not a callback."""
    now = now or _now()
    started = time.monotonic()

    db = _connect(client, logger, sleep)

    logger.info(
        "run_start",
        trigger="scheduler",
        windowHours=config.CLUSTER_WINDOW_HOURS,
        source="mongo",
        gpu=config.CLUSTERING_GPU,
        # The ping above left a pooled connection; expect "zstd" when MONGODB_COMPRESSORS=zstd.
        mongoCompressor=negotiated_compressor(client),
    )

    fetch_start = time.monotonic()
    try:
        window = reader.read(db, now)
    except Exception as err:  # noqa: BLE001 - window read failure is infra-fatal
        raise InfraFatal(f"window read failed: {err}") from err
    fetch_ms = int((time.monotonic() - fetch_start) * 1000)
    window_count = len(window)
    logger.info(
        "window_fetched",
        articleCount=window_count,
        windowStart=(now - timedelta(hours=config.CLUSTER_WINDOW_HOURS)).isoformat(),
        windowEnd=now.isoformat(),
        fetchMs=fetch_ms,
        **_mem_fields(),
    )

    clustering_run_id = str(uuid.uuid4())
    cluster_count = 0
    cluster_duration_ms = 0
    fit_ms = 0
    persist_ms = 0
    stable_id_ms = 0
    cluster_docs = 0
    link_docs = 0
    multi = 0
    singletons = 0
    stable_carried = 0
    stable_minted = 0
    max_cluster_size = 0
    mean_cluster_size = 0.0
    p99_cluster_size = 0

    if window:
        cluster_start = time.monotonic()
        # The fit and persist need only the ids and one float32 matrix. Drop the
        # docs (and their per-doc vectors) before the fit so the window is never
        # held twice in host memory.
        article_ids = [doc["_id"] for doc in window]
        vectors = np.asarray([doc[WINDOW_VECTOR_FIELD] for doc in window], dtype=np.float32)
        del window
        prep_ms = int((time.monotonic() - cluster_start) * 1000)
        labels, probs = _fit(config, logger, cluster_svc, vectors, prep_ms)
        del vectors  # normalized in place by the fit; nothing reads it after
        labels = uniquify_noise_labels(labels)
        sizes = cluster_sizes(labels)
        multi = sum(1 for label, n in sizes.items() if label >= 0 and n >= 2)
        singletons = sum(1 for label in sizes if label < 0)
        fit_ms = int((time.monotonic() - cluster_start) * 1000)
        # Size-distribution instrument for cluster_selection_epsilon changes —
        # computed from `sizes` above, BEFORE persist's PERSIST_SINGLETON_CLUSTERS
        # filter, so the fields mean the same thing whether that filter is on or
        # off: the shape HDBSCAN produced, not what got persisted.
        # maxClusterSize is the largest cluster in the whole window; mean and
        # p99 are over multi-member clusters. p99 is cluster-weighted (one entry per
        # cluster, not per article), so a single runaway blob is one outlier
        # among many thousands of clusters and will NOT clear the 99th
        # percentile — maxClusterSize is the blob detector; p99ClusterSize
        # instead catches broad tail inflation (many clusters growing at once).
        multi_sizes = [n for label, n in sizes.items() if label >= 0 and n >= 2]
        if multi_sizes:
            max_cluster_size = max(multi_sizes)
            mean_cluster_size = round(sum(multi_sizes) / len(multi_sizes), 2)
            p99_cluster_size = int(np.percentile(multi_sizes, 99))
        logger.info(
            "hdbscan_fit",
            clusterCount=multi,
            noiseCount=singletons,
            noisePct=round(100.0 * singletons / window_count, 1),
            fitMs=fit_ms,
            **_mem_fields(),
        )
        clusters = build_clusters(article_ids, labels, probs)
        del article_ids

        # Stable cluster identity — carry a stableClusterId across the persist wipe
        # for multi-member clusters (size >= 2). The previous generation MUST be
        # read here, strictly before persist's delete_many; after the wipe both
        # collections are empty and everything would re-mint. Singletons carry no
        # stableClusterId (their identity is the article id).
        stable_id_start = time.monotonic()
        prev_membership, prev_stable_ids = read_prev(db)
        multi_clusters = [c for c in clusters if len(c["members"]) >= 2]
        member_id_lists = [[aid for aid, _ in c["members"]] for c in multi_clusters]
        match = match_stable_ids(
            member_id_lists,
            prev_membership,
            prev_stable_ids,
            min_shared=config.STABLE_MIN_SHARED,
            min_overlap_ratio=config.STABLE_MIN_OVERLAP_RATIO,
        )
        for cluster, stable_id in zip(multi_clusters, match.stable_ids):
            cluster["stableClusterId"] = stable_id
        stable_carried = match.carried
        stable_minted = match.minted
        stable_id_ms = int((time.monotonic() - stable_id_start) * 1000)
        logger.info(
            "stable_ids_assigned",
            multiClusters=len(multi_clusters),
            prevStableClusters=len(prev_stable_ids),
            stableCarried=stable_carried,
            stableMinted=stable_minted,
            stableIdMs=stable_id_ms,
        )
        # A previous generation existed but nothing carried — the identity chain
        # broke (mass re-mint). On the first run after deploy prev is empty, so
        # this stays quiet (all-mint is expected there, logged at info above).
        if prev_stable_ids and stable_carried == 0:
            logger.warn(
                "stable_ids_reminted",
                prevStableClusters=len(prev_stable_ids),
                multiClusters=len(multi_clusters),
                stableMinted=stable_minted,
            )

        persist_start = time.monotonic()
        try:
            result = persist(
                db, clusters, config,
                clustering_run_id=clustering_run_id, now=now, logger=logger,
            )
        except Exception as err:  # noqa: BLE001 - persistence failure is infra-fatal
            raise InfraFatal(f"cluster persist failed: {err}") from err
        cluster_count = result["clusters"]
        cluster_docs = result["clusterDocs"]
        link_docs = result["linkDocs"]
        persist_ms = int((time.monotonic() - persist_start) * 1000)
        cluster_duration_ms = int((time.monotonic() - cluster_start) * 1000)
        logger.info(
            "clusters_persisted",
            clusterDocs=cluster_docs,
            linkDocs=link_docs,  # actual docs written (singleton skip aware)
            persistMs=persist_ms,
            **_mem_fields(),
        )
        # Stamping (processingStatus="aggregation_successful") happens inside
        # persist_clusters on the same article set — surfaced here as its own stage.
        logger.info(
            "articles_stamped",
            stampedCount=window_count,
            totalMs=int((time.monotonic() - started) * 1000),
        )

    summary = {
        "job": CURRENT_JOB,
        "windowArticles": window_count,
        "clusters": cluster_count,
        "multiArticleClusters": multi,
        "singletons": singletons,
        "stableCarried": stable_carried,
        "stableMinted": stable_minted,
        "maxClusterSize": max_cluster_size,
        "p99ClusterSize": p99_cluster_size,
        "meanClusterSize": mean_cluster_size,
        "clusteringRunId": clustering_run_id,
        "clusterDurationMs": cluster_duration_ms,
        "fetchMs": fetch_ms,
        "fitMs": fit_ms,
        "persistMs": persist_ms,
        "stableIdMs": stable_id_ms,
        "durationMs": int((time.monotonic() - started) * 1000),
        **_mem_fields(),
    }
    logger.info("run_summary", **summary)
    return summary


def _fit(config: Settings, logger: JobLogger, cluster_svc: ClusterService, vectors: np.ndarray, prep_ms: int):
    """Fit HDBSCAN on the window's float32 text-matching matrix (consumed: it is
    normalized in place). A window of <2 vectors can't form a cluster
    (min_cluster_size=2), so every article is a noise singleton — skip the
    hdbscan / GPU load entirely."""
    n = len(vectors)
    if n < 2:
        return np.full(n, -1, dtype=int), np.ones(n, dtype=float)
    try:
        cluster_svc.load(logger)  # lazy hdbscan / GPU init — infra-fatal on failure
        logger.info(
            "matrix_built",
            rows=int(vectors.shape[0]),
            dims=int(vectors.shape[1]),
            device="gpu" if config.CLUSTERING_GPU else "cpu",
            prepMs=prep_ms,
            **_mem_fields(),
        )
        return cluster_svc.cluster(vectors)
    except Exception as err:  # noqa: BLE001
        raise InfraFatal(f"clustering init/fit failed: {err}") from err


# ---------------------------------------------------------------------------
# Composition root — builds the real pymongo client.
# ---------------------------------------------------------------------------

def _build_client(config: Settings):
    import pymongo

    kwargs = dict(
        maxPoolSize=config.MONGODB_MAX_POOL_SIZE,
        minPoolSize=0,
        maxIdleTimeMS=30_000,
        serverSelectionTimeoutMS=10_000,
        appname=CURRENT_JOB,
        readPreference=config.MONGODB_READ_PREFERENCE,
    )
    # Wire compression. Omitted entirely when the setting is empty so the driver
    # default (no compression) applies.
    if config.MONGODB_COMPRESSORS:
        kwargs["compressors"] = config.MONGODB_COMPRESSORS
    # pymongo silently drops a compressor it cannot load (zstd below Python 3.14
    # needs backports.zstd), leaving only a warning. Fail the bootstrap instead.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client = pymongo.MongoClient(config.MONGODB_URI, **kwargs)
    dropped = [str(w.message) for w in caught if "compress" in str(w.message).lower()]
    if dropped:
        client.close()
        raise RuntimeError(f"MONGODB_COMPRESSORS={config.MONGODB_COMPRESSORS!r} unusable: {dropped[0]}")
    return client


def negotiated_compressor(client) -> str | None:
    """The wire compressor a pooled connection actually negotiated with the server
    (None = uncompressed). shortcut: reads pymongo 4.19 internals and reports
    "unknown" if they move; replace if pymongo ever exposes this publicly."""
    try:
        for server in client._topology._servers.values():
            for conn in server.pool.conns:
                ctx = conn.compression_context
                return type(ctx).__name__.removesuffix("Context").lower() if ctx else None
    except Exception:  # noqa: BLE001 - instrumentation must never fail a run
        pass
    return "unknown"


def main() -> int:
    config = Settings()
    logger = JobLogger(CURRENT_JOB, level=config.LOG_LEVEL)
    logger.info(
        "process_started",
        importToMainMs=int((time.monotonic() - _IMPORT_TS) * 1000),
    )

    # One execution per job at a time. Cloud Run
    # has no concurrencyPolicy; the oldest execution wins, the rest yield.
    decision = singleton_decision(config.CHAIN_GCP_PROJECT, config.CHAIN_GCP_REGION, logger)
    if not decision.run:
        logger.info("singleton_skip", yieldedTo=decision.blocker)
        return 0

    try:
        client = _build_client(config)
    except Exception as err:  # noqa: BLE001 - Mongo config unusable at bootstrap
        logger.error("infra_fatal", stage="bootstrap", error=str(err))
        return 1

    reader = MongoWindowReader(config)
    cluster_svc = ClusterService(config)

    try:
        run_clustering_mongo(config, logger, client, reader, cluster_svc)
        return 0
    except InfraFatal as err:
        logger.error("infra_fatal", stage="run", error=str(err))
        return 1
    except Exception as err:  # noqa: BLE001 - anything unexpected is infra-fatal
        logger.error("infra_fatal", stage="run", error=repr(err))
        return 1
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
