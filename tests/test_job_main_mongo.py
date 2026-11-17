"""Unit tests for src/job_main_mongo.py pure helpers. The run itself is covered
by the integration tests against a real mongod."""
from __future__ import annotations

import numpy as np
import pytest
from bson import ObjectId

from src.job_main_mongo import (
    build_clusters,
)


# --------------------------------------------------------------------------- #
# build_clusters
# --------------------------------------------------------------------------- #
class TestBuildClusters:
    def test_groups_by_label(self):
        w = [ObjectId(), ObjectId(), ObjectId()]
        labels = np.array([0, 0, -1])
        probs = np.array([0.9, 0.8, 0.0])
        clusters = build_clusters(w, labels, probs)
        # one 2-member cluster (label 0) + one singleton (label -1)
        sizes = sorted(len(c["members"]) for c in clusters)
        assert sizes == [1, 2]

    def test_noise_members_confidence_one(self):
        w = [ObjectId()]
        clusters = build_clusters(w, np.array([-1]), np.array([0.0]))
        assert clusters[0]["members"][0][1] == 1.0

    def test_real_cluster_uses_probability(self):
        w = [ObjectId(), ObjectId()]
        clusters = build_clusters(w, np.array([0, 0]), np.array([0.7, 0.6]))
        confs = sorted(m[1] for m in clusters[0]["members"])
        assert confs == pytest.approx([0.6, 0.7])
