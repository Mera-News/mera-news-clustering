"""Unit tests for src/services/cluster_service.py: the pure helpers and one real
CPU hdbscan fit on a tiny matrix."""
from __future__ import annotations


import numpy as np
import pytest

from src.config import Settings
from src.services.cluster_service import (
    ClusterService,
    _to_host,
    cluster_sizes,
    l2_normalize,
    uniquify_noise_labels,
)


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


def _svc(**overrides) -> ClusterService:
    return ClusterService(_settings(**overrides))


# --------------------------------------------------------------------------- #
# _to_host
# --------------------------------------------------------------------------- #
class TestToHost:
    def test_get_branch(self):
        inner = np.array([1.0])

        class _Cupy:
            def get(self):
                return inner

        assert _to_host(_Cupy()) is inner

    def test_plain_ndarray(self):
        arr = np.array([1.0, 2.0])
        np.testing.assert_array_equal(_to_host(arr), arr)

    def test_plain_list(self):
        assert isinstance(_to_host([1.0]), np.ndarray)


# --------------------------------------------------------------------------- #
# l2_normalize
# --------------------------------------------------------------------------- #
class TestL2Normalize:
    def test_unit_length(self):
        out = l2_normalize(np.array([[3.0, 4.0]]))
        assert float(np.linalg.norm(out[0])) == pytest.approx(1.0)

    def test_zero_row_stays_finite(self):
        out = l2_normalize(np.array([[0.0, 0.0, 0.0]]))
        assert np.all(np.isfinite(out))

    def test_in_place_reuses_float32_buffer_with_same_values(self):
        x = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 2.0]], dtype=np.float32)
        expected = l2_normalize(x)
        out = l2_normalize(x, in_place=True)
        assert out is x
        assert np.array_equal(out, expected)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            l2_normalize(np.array([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- #
# cluster_sizes / uniquify_noise_labels (pure label helpers)
# --------------------------------------------------------------------------- #
class TestClusterSizes:
    def test_noise_is_size_one_each(self):
        sizes = cluster_sizes(np.array([0, 0, -1, -2]))
        assert sizes[0] == 2
        assert sizes[-1] == 1
        assert sizes[-2] == 1

    def test_single_cluster(self):
        assert cluster_sizes(np.array([3, 3, 3]))[3] == 3


class TestUniquifyNoiseLabels:
    def test_each_noise_point_gets_unique_negative_label(self):
        labels = uniquify_noise_labels(np.array([0, -1, 0, -1, -1]))
        assert list(labels) == [0, -1, 0, -2, -3]

    def test_no_noise_is_identity(self):
        labels = uniquify_noise_labels(np.array([0, 1, 1, 0]))
        assert list(labels) == [0, 1, 1, 0]

    def test_all_noise_all_unique(self):
        labels = uniquify_noise_labels(np.array([-1, -1, -1]))
        assert sorted(labels) == [-3, -2, -1]
        assert len(set(labels.tolist())) == 3


# --------------------------------------------------------------------------- #
# Real hdbscan smoke test (proves the CPU path actually clusters)
# --------------------------------------------------------------------------- #
class TestRealHDBSCAN:
    def test_two_identical_vectors_cluster_together(self):
        svc = _svc()
        svc.load()  # real hdbscan import
        # Two tight groups of 2 identical unit vectors → two clusters, no noise.
        vecs = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32
        )
        labels, probs = svc.cluster(vecs)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
        assert len(probs) == 4
