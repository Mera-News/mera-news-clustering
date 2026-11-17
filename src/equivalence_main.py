"""Dependency-equivalence fit: same frozen window, same params, two images.

Answers "did a dependency bump change the clustering?" without a labelled truth
set. Run it under the old image and the new one with the SAME
`EQUIV_WINDOW_END`; each run writes its labels to the scratch collection
`ml-equivalence-run`. A third run with
`EQUIV_COMPARE=<tagA>,<tagB>` compares two stored runs and logs
`equivalence_verdict`; `EQUIV_DROP=true` empties the collection. Every mode
runs inside the job, so it works wherever the job can reach Mongo. Never
persists clusters, never stamps articles, skips the singleton guard (it touches
nothing the job owns).

    JOB_ENTRYPOINT=src.equivalence_main EQUIV_TAG=new-gpu \
    EQUIV_WINDOW_END=2026-10-31T12:00:00+00:00 python -m cuml.accel -m src.equivalence_main

Run it against a copy of the data, not the live database. Each fit consumes a
copy of the matrix (ClusterService normalizes in place), so peak host RAM is
about twice the job's.
"""
from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime
from importlib import metadata

import numpy as np
from bson import Binary

from src.config import Settings
from src.job_logger import JobLogger
from src.job_main_mongo import _build_client, _connect, _mem_fields
from src.mongo.window_reader import WINDOW_VECTOR_FIELD, MongoWindowReader
from src.services.cluster_service import ClusterService

JOB = "news-clustering-equivalence"
COLLECTION = "ml-equivalence-run"


class EquivSettings(Settings):
    # Fit mode needs both; compare and drop modes need neither.
    EQUIV_TAG: str = ""
    EQUIV_WINDOW_END: datetime | None = None
    EQUIV_COMPARE: str = ""
    EQUIV_DROP: bool = False
    # 0 = the whole window. A CPU run needs a cap (CPU HDBSCAN is superlinear).
    EQUIV_MAX_ROWS: int = 0
    EQUIV_SEED: int = 17
    # 0.0 is HDBSCAN's default; 0.30 is the value Mera runs.
    EQUIV_EPSILONS: str = "0,0.30"


def canonical_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters by first appearance (noise stays -1), so two fits that
    produce the same partition produce the same array."""
    out = np.full(len(labels), -1, dtype=np.int32)
    seen: dict[int, int] = {}
    for i, value in enumerate(labels.tolist()):
        if value >= 0:
            out[i] = seen.setdefault(value, len(seen))
    return out


def select_rows(n: int, max_rows: int, seed: int) -> np.ndarray:
    if not max_rows or n <= max_rows:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for dist in ("numpy", "scikit-learn", "hdbscan", "scipy", "pymongo", "cuml-cu12"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = None
    return out


def run_equivalence(config: EquivSettings, logger: JobLogger, db, reader, make_service=ClusterService) -> list[dict]:
    window = reader.read(db, config.EQUIV_WINDOW_END)
    ids = [str(d["_id"]) for d in window]
    vectors = np.asarray([d[WINDOW_VECTOR_FIELD] for d in window], dtype=np.float32)
    del window
    rows = select_rows(len(ids), config.EQUIV_MAX_ROWS, config.EQUIV_SEED)
    ids = [ids[i] for i in rows]
    vectors = vectors[rows]
    ids_digest = _digest("\n".join(ids).encode())
    logger.info("equivalence_window", windowEnd=config.EQUIV_WINDOW_END.isoformat(),
                rows=len(ids), idsDigest=ids_digest, **_mem_fields())

    results = []
    for eps in (float(e) for e in config.EQUIV_EPSILONS.split(",")):
        svc = make_service(config.model_copy(update={"CLUSTER_SELECTION_EPSILON": eps}))
        svc.load(logger)
        started = time.monotonic()
        labels, probs = svc.cluster(vectors.copy())
        fit_ms = int((time.monotonic() - started) * 1000)
        canon = canonical_labels(np.asarray(labels))
        sizes = np.bincount(canon[canon >= 0]) if (canon >= 0).any() else np.zeros(0, dtype=int)
        doc = {
            "tag": config.EQUIV_TAG,
            "epsilon": eps,
            "windowEnd": config.EQUIV_WINDOW_END,
            "rows": len(ids),
            "idsDigest": ids_digest,
            "labelsDigest": _digest(canon.tobytes()),
            "clusters": int(len(sizes)),
            "noise": int((canon < 0).sum()),
            "maxClusterSize": int(sizes.max()) if len(sizes) else 0,
            "fitMs": fit_ms,
            "cumlLoaded": "cuml" in sys.modules,
            "versions": _versions(),
            "labels": Binary(canon.tobytes()),
            "probabilities": Binary(np.asarray(probs, dtype=np.float32).tobytes()),
        }
        db[COLLECTION].replace_one({"tag": config.EQUIV_TAG, "epsilon": eps}, doc, upsert=True)
        logger.info("equivalence_fit", **{k: v for k, v in doc.items() if k not in ("labels", "probabilities")},
                    **_mem_fields())
        results.append(doc)
    return results


def _with_unique_noise(labels: np.ndarray) -> np.ndarray:
    """Noise (-1) is not one cluster: give each noise row its own label for ARI."""
    out = labels.astype(np.int64)
    noise = out < 0
    out[noise] = -np.arange(1, noise.sum() + 1)
    return out


def compare_runs(a: dict, b: dict) -> dict:
    """Identical partition or not; when not, the ARI and size figures the owner judges by."""
    if a["idsDigest"] != b["idsDigest"]:
        return {"identical": False, "error": "different input rows (window or row cap differs)"}
    same = a["labelsDigest"] == b["labelsDigest"]
    report = {
        "identical": same,
        "maxClusterSize": [a["maxClusterSize"], b["maxClusterSize"]],
        "clusters": [a["clusters"], b["clusters"]],
        "noise": [a["noise"], b["noise"]],
        "fitMs": [a["fitMs"], b["fitMs"]],
        "cumlLoaded": [a["cumlLoaded"], b["cumlLoaded"]],
        "versions": [a["versions"], b["versions"]],
    }
    pa = np.frombuffer(a["probabilities"], dtype=np.float32)
    pb = np.frombuffer(b["probabilities"], dtype=np.float32)
    report["probMaxAbsDiff"] = float(np.max(np.abs(pa - pb))) if len(pa) else 0.0
    if same:
        report["ari"] = 1.0
    else:
        from sklearn.metrics import adjusted_rand_score

        la = np.frombuffer(a["labels"], dtype=np.int32)
        lb = np.frombuffer(b["labels"], dtype=np.int32)
        report["ari"] = round(float(adjusted_rand_score(_with_unique_noise(la), _with_unique_noise(lb))), 6)
    return report


def run_compare(config: EquivSettings, logger: JobLogger, db) -> bool:
    tag_a, tag_b = (t.strip() for t in config.EQUIV_COMPARE.split(","))
    runs = {tag: {d["epsilon"]: d for d in db[COLLECTION].find({"tag": tag})} for tag in (tag_a, tag_b)}
    shared = sorted(set(runs[tag_a]) & set(runs[tag_b]))
    ok = bool(shared)
    for eps in shared:
        report = compare_runs(runs[tag_a][eps], runs[tag_b][eps])
        ok &= report["identical"]
        logger.info("equivalence_compare", tags=[tag_a, tag_b], epsilon=eps,
                    rows=runs[tag_a][eps]["rows"], **report)
    logger.info("equivalence_verdict", tags=[tag_a, tag_b], epsilons=shared,
                verdict="IDENTICAL" if ok else "DIFFERENT")
    return ok


def main() -> int:
    config = EquivSettings()
    logger = JobLogger(JOB, level=config.LOG_LEVEL)
    client = _build_client(config)
    try:
        db = _connect(client, logger, time.sleep)
        if config.EQUIV_DROP:
            logger.info("equivalence_dropped", deleted=db[COLLECTION].delete_many({}).deleted_count)
        elif config.EQUIV_COMPARE:
            # A DIFFERENT verdict is a finding for the owner, not a job failure.
            run_compare(config, logger, db)
        else:
            if not config.EQUIV_TAG or config.EQUIV_WINDOW_END is None:
                raise ValueError("fit mode needs EQUIV_TAG and EQUIV_WINDOW_END")
            run_equivalence(config, logger, db, MongoWindowReader(config))
        return 0
    except Exception as err:  # noqa: BLE001 - a failed check run exits non-zero
        logger.error("infra_fatal", stage="equivalence", error=repr(err))
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
