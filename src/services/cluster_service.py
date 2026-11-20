"""HDBSCAN clustering over the 48h embedding window.

Written against the plain CPU `hdbscan` package API (`hdbscan.HDBSCAN`). The SAME
code runs on GPU with zero changes when launched under `python -m cuml.accel`
(RAPIDS zero-code-change mode intercepts the `hdbscan` import and routes it to
cuML/GPU). Launched plainly (`python -m src.job_main_mongo`, no cuml.accel, or no
GPU) it runs on CPU — which is what lets this container run in CI and the dev
harness on small fixtures with no GPU.

Params: L2-normalized vectors, euclidean metric (≡ cosine on
unit vectors), min_cluster_size=2, min_samples=1, cluster_selection_method="eom".

This service is pure compute (numpy + hdbscan) — no Mongo, no IO. The
job_main_mongo composition root feeds it a vector matrix and reads back its labels.
"""
from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from typing import Any

import numpy as np

from src.config import Settings
from src.job_logger import JobLogger

logger = logging.getLogger(__name__)


def uniquify_noise_labels(labels: np.ndarray) -> np.ndarray:
    """HDBSCAN stamps EVERY noise point with the shared label -1, but downstream
    cluster identity is keyed on (clusteringRunId, clusterLabel) — a shared -1
    would silently merge all unrelated noise singletons in a run into one cluster
    doc. Real labels (>= 0) pass through; the i-th noise point becomes -(i+1),
    giving each singleton its own identity."""
    out = labels.copy()
    next_label = -1
    for i, value in enumerate(labels):
        if int(value) == -1:
            out[i] = next_label
            next_label -= 1
    return out


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Cluster size per (uniquified) label. Negative labels are noise
    singletons (size 1 by construction); real clusters are the count of
    members with that label."""
    counts = Counter(int(x) for x in labels)
    return {label: (1 if label < 0 else n) for label, n in counts.items()}


def _to_host(arr: Any) -> np.ndarray:
    """Convert a fit output (numpy on CPU, or cupy under cuml.accel) to numpy."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def l2_normalize(vectors: np.ndarray, *, in_place: bool = False) -> np.ndarray:
    """L2-normalize rows so euclidean distance ≡ cosine distance. A zero row is
    left as zeros (its norm is forced to 1 to avoid NaN). `in_place=True`
    divides a float32 input in place (same values, no second window-sized
    matrix); other dtypes still get one float32 copy."""
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    if in_place:
        x /= norms
        return x
    return x / norms


class ClusterService:
    """Loads `hdbscan.HDBSCAN` lazily on `load()`, then `cluster()` L2-normalizes
    the vector matrix and fits HDBSCAN, returning `(labels, probabilities)` as
    host numpy arrays. Noise points carry HDBSCAN's shared label -1; the caller
    (uniquify_noise_labels, above) remaps each to a unique negative label so
    every article ends up in exactly one distinct cluster."""

    def __init__(self, config: Settings):
        self.config = config
        self._hdbscan_cls: Any = None
        self._ready = False

    def load(self, job_logger: JobLogger | None = None) -> None:
        # Lazy import so CPU/test paths stay light and cuml.accel can intercept
        # the `hdbscan` import when the job is launched under it. An import
        # failure here is infra-fatal (bad image / GPU init) — it propagates.
        import hdbscan

        # Provenance: which hdbscan module resolved, and whether cuml is loaded —
        # under `python -m cuml.accel` cuml.accel intercepts the hdbscan import
        # and routes the fit to the GPU. Logging the module path + cuml presence
        # proves (in the job's logs) that interception actually happened. Emitted via
        # the structured JobLogger (JSON to stdout) — the stdlib `logger` above
        # has no configured handler in this job, so those lines never reach
        # Cloud Logging. `job_logger` defaults to None so callers/tests that
        # don't wire one (e.g. direct `ClusterService.load()` calls) still work.
        if job_logger is not None:
            job_logger.info(
                "hdbscan_provenance",
                hdbscanModule=getattr(hdbscan, "__file__", "?"),
                cumlLoaded="cuml" in sys.modules,
            )

        self._hdbscan_cls = hdbscan.HDBSCAN
        self._ready = True
        logger.info(
            "ClusterService ready (min_cluster_size=%d, min_samples=%d, method=%s)",
            self.config.MIN_CLUSTER_SIZE,
            self.config.MIN_SAMPLES,
            self.config.CLUSTER_SELECTION_METHOD,
        )

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _kwargs(self) -> dict:
        kwargs = dict(
            min_cluster_size=self.config.MIN_CLUSTER_SIZE,
            min_samples=self.config.MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method=self.config.CLUSTER_SELECTION_METHOD,
        )
        if self.config.CLUSTER_SELECTION_EPSILON > 0:
            kwargs["cluster_selection_epsilon"] = self.config.CLUSTER_SELECTION_EPSILON
        return kwargs

    def cluster(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit HDBSCAN on the (L2-normalized) vectors. Returns
        `(labels, probabilities)` — int labels (-1 = noise) and float membership
        confidences in [0, 1], both host numpy arrays aligned to input rows.
        Consumes `vectors`: a float32 matrix is normalized in place."""
        if not self._ready:
            raise RuntimeError("ClusterService not loaded")

        x = l2_normalize(vectors, in_place=True)
        n = x.shape[0]

        fit_start = time.monotonic()
        model = self._hdbscan_cls(**self._kwargs())
        labels = _to_host(model.fit_predict(x)).astype(int)

        probs = getattr(model, "probabilities_", None)
        if probs is not None:
            probabilities = _to_host(probs).astype(float)
        else:
            probabilities = np.ones(n, dtype=float)

        logger.info("HDBSCAN fit in %.2fs (%d vectors)", time.monotonic() - fit_start, n)
        return labels, probabilities
