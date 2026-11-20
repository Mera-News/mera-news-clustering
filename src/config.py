"""Runtime configuration for the news-clustering job (Mongo in → Mongo out).

The clustering job reads a 48h window of `news-article` embeddings from Mongo,
fits HDBSCAN, and writes clusters back to Mongo — pure Mongo↔Mongo. Finalization
(`link-all-topics-to-clusters`) is enqueued by a separate scheduler cron, not by
this job.

All settings come from env (Pydantic Settings, case-sensitive).
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Logging -----------------------------------------------------------
    # Levels are the alerting contract (same as the data server's JobLogger): info =
    # cumulative run narrative, warn = dropped/skip, error = infra-fatal. No debug.
    LOG_LEVEL: str = "info"

    # --- HDBSCAN params ----------------------------------------------------
    # L2-normalized vectors + euclidean ≡ cosine; min_cluster_size=2, min_samples=1,
    # cluster_selection_method="eom". Same code runs CPU (`python -m src.job_main_mongo`)
    # or GPU (`python -m cuml.accel -m src.job_main_mongo`, zero-code-change acceleration).
    MIN_CLUSTER_SIZE: int = 2
    MIN_SAMPLES: int = 1
    CLUSTER_SELECTION_METHOD: str = "eom"
    CLUSTER_SELECTION_EPSILON: float = 0.0
    # Informational only. The actual accelerator comes from launching under
    # `python -m cuml.accel`; this flag is surfaced in logs and lets an operator
    # force the CPU path even on a GPU host. It does NOT itself load a GPU.
    CLUSTERING_GPU: bool = False

    # --- Cloud Run singleton guard (src/singleton.py) ----------------------
    # The guard lists this job's executions via the run.jobs REST API to enforce
    # one execution at a time. project/region address that API; empty outside
    # Cloud Run (guard becomes a no-op). The CHAIN_* names are kept for existing
    # deployments.
    CHAIN_GCP_PROJECT: str = ""
    CHAIN_GCP_REGION: str = "europe-west1"

    # --- Mongo IO (src/job_main_mongo.py) ----------------------------------
    # Reads embeddings from Mongo, clusters, and writes clusters back to Mongo —
    # pure Mongo↔Mongo, no callback.
    MONGODB_URI: str = ""
    MONGODB_MAX_POOL_SIZE: int = 1
    MONGODB_READ_PREFERENCE: str = "primary"
    # Wire compression for the pymongo client. "zstd" uses Python 3.14's stdlib
    # `compression.zstd` (no package); the job refuses to start if the driver
    # cannot do it. Empty string disables compression (no `compressors` kwarg).
    MONGODB_COMPRESSORS: str = "zstd"

    # Clustering window: how far back the Mongo window reader pulls embeddings.
    CLUSTER_WINDOW_HOURS: int = 48

    # --- Window fetch tuning (src/mongo/window_reader.py) ------------------
    # Cursor batch size for the window read (larger batch = fewer getMore
    # round-trips over the wire). FETCH_PARALLELISM > 1 splits [cutoff, now] into
    # N equal createdAt sub-ranges fetched on a thread pool then concatenated.
    # Measured on 22.6k articles (2026-07-20): 1->14.2s, 2->6.5s, 4->6.0s,
    # 8->15.1s (regresses — BSON-decode/cursor contention); parallel fetch
    # saturates ~4, so that's the default. Primary reads also measured faster
    # than secondaryPreferred at parallelism 1 (14.2s vs 16.3s), and keep this
    # heavy hourly scan off the mongot-hosting secondaries (see
    # MONGODB_READ_PREFERENCE above).
    FETCH_BATCH_SIZE: int = 2000
    FETCH_PARALLELISM: int = 4

    # --- DB write tuning (src/mongo/persist_clusters.py) -------------------
    # Persist inserts are UNPACED and PARALLEL by default: chunk each collection's
    # docs at DB_WRITE_BATCH_SIZE and fan the chunks across a thread pool of
    # min(DB_WRITE_CONCURRENCY, MONGODB_MAX_POOL_SIZE) workers (raise
    # MONGODB_MAX_POOL_SIZE for the concurrency to actually take effect). The old
    # pacing (256-doc chunks + a 0.15s inter-chunk sleep) is gone; a
    # non-zero DB_WRITE_PAUSE_S is retained ONLY as an emergency throttle that
    # forces the sequential paced fallback.
    DB_WRITE_BATCH_SIZE: int = 2000
    DB_WRITE_PAUSE_S: float = 0.0
    DB_WRITE_CONCURRENCY: int = 4
    # Chunk size for the filtered article stamp (update_many on the window ids).
    STAMP_BATCH_SIZE: int = 10_000
    # When False, size-1 (singleton/noise) clusters get NO cluster doc and NO
    # link docs — every window article is still stamped. When True (default),
    # singletons are materialized as size-1 cluster docs (the original behavior).
    PERSIST_SINGLETON_CLUSTERS: bool = True
    # When True, restore the original wipe-first order (delete_many({}) both
    # collections, then insert, then stamp). Default False uses a generation swap
    # (insert the new generation, then delete every doc from prior generations by
    # clusteringRunId) so readers never observe an empty snapshot mid-run.
    PERSIST_LEGACY_WIPE_FIRST: bool = False

    # --- Stable cluster identity (membership-overlap carry across runs) -----
    # Multi-member clusters (size >= 2) carry a `stableClusterId` across runs by
    # matching membership overlap against the previous generation. A NEW cluster
    # inherits a previous cluster's stableClusterId when it shares at least
    # STABLE_MIN_SHARED member articles AND that overlap is at least
    # STABLE_MIN_OVERLAP_RATIO of the smaller of the two clusters' sizes (greedy
    # one-to-one, highest overlap wins). Singletons carry no stableClusterId —
    # their identity is the article id.
    STABLE_MIN_SHARED: int = 2
    STABLE_MIN_OVERLAP_RATIO: float = 0.5

    model_config = {"case_sensitive": True}
