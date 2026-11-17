"""Unit tests for src/config.py — news-clustering-job Settings."""
from __future__ import annotations

import pytest

from src.config import Settings


def _s(**overrides) -> Settings:
    return Settings(**overrides)


class TestDefaults:
    def test_log_level_default(self):
        assert _s().LOG_LEVEL == "info"

    def test_min_cluster_size_default(self):
        assert _s().MIN_CLUSTER_SIZE == 2

    def test_min_samples_default(self):
        assert _s().MIN_SAMPLES == 1

    def test_cluster_selection_method_default(self):
        assert _s().CLUSTER_SELECTION_METHOD == "eom"

    def test_cluster_selection_epsilon_default(self):
        assert _s().CLUSTER_SELECTION_EPSILON == 0.0

    def test_clustering_gpu_default_false(self):
        assert _s().CLUSTERING_GPU is False

    def test_chain_gcp_region_default(self):
        assert _s().CHAIN_GCP_REGION == "europe-west1"

    def test_chain_gcp_project_default_empty(self):
        assert _s().CHAIN_GCP_PROJECT == ""

    def test_cluster_window_hours_default(self):
        assert _s().CLUSTER_WINDOW_HOURS == 48

    def test_mongodb_defaults(self):
        s = _s()
        assert s.MONGODB_URI == ""
        assert s.MONGODB_MAX_POOL_SIZE == 1
        assert s.MONGODB_READ_PREFERENCE == "primary"
        assert s.MONGODB_COMPRESSORS == "zstd"

    def test_db_write_defaults_unpaced_parallel(self):
        s = _s()
        # Rewrite defaults: bigger batches, no pacing pause, parallel fan-out.
        assert s.DB_WRITE_BATCH_SIZE == 2000
        assert s.DB_WRITE_PAUSE_S == pytest.approx(0.0)
        assert s.DB_WRITE_CONCURRENCY == 4
        assert s.STAMP_BATCH_SIZE == 10_000

    def test_persist_flag_defaults(self):
        s = _s()
        assert s.PERSIST_SINGLETON_CLUSTERS is True
        assert s.PERSIST_LEGACY_WIPE_FIRST is False

    def test_fetch_tuning_defaults(self):
        s = _s()
        assert s.FETCH_BATCH_SIZE == 2000
        assert s.FETCH_PARALLELISM == 4


class TestEnvOverrides:
    def test_string_field_env(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "warn")
        assert _s().LOG_LEVEL == "warn"

    def test_int_field_env(self, monkeypatch):
        monkeypatch.setenv("CLUSTER_WINDOW_HOURS", "72")
        assert _s().CLUSTER_WINDOW_HOURS == 72

    def test_float_field_env(self, monkeypatch):
        monkeypatch.setenv("CLUSTER_SELECTION_EPSILON", "0.25")
        assert _s().CLUSTER_SELECTION_EPSILON == pytest.approx(0.25)

    def test_bool_field_env(self, monkeypatch):
        monkeypatch.setenv("CLUSTERING_GPU", "true")
        assert _s().CLUSTERING_GPU is True

    def test_persist_legacy_wipe_first_env(self, monkeypatch):
        monkeypatch.setenv("PERSIST_LEGACY_WIPE_FIRST", "true")
        assert _s().PERSIST_LEGACY_WIPE_FIRST is True

    def test_fetch_parallelism_env(self, monkeypatch):
        monkeypatch.setenv("FETCH_PARALLELISM", "8")
        assert _s().FETCH_PARALLELISM == 8

    def test_compressors_env_empty(self, monkeypatch):
        monkeypatch.setenv("MONGODB_COMPRESSORS", "")
        assert _s().MONGODB_COMPRESSORS == ""


class TestCaseSensitivity:
    def test_lowercase_env_ignored(self, monkeypatch):
        monkeypatch.setenv("log_level", "warn")
        assert _s().LOG_LEVEL == "info"


class TestTypeCoercion:
    def test_int_coercion(self, monkeypatch):
        monkeypatch.setenv("DB_WRITE_BATCH_SIZE", "250")
        s = _s()
        assert isinstance(s.DB_WRITE_BATCH_SIZE, int)
        assert s.DB_WRITE_BATCH_SIZE == 250

    def test_kwargs_override(self):
        s = _s(CLUSTER_WINDOW_HOURS=24, MIN_CLUSTER_SIZE=5)
        assert s.CLUSTER_WINDOW_HOURS == 24
        assert s.MIN_CLUSTER_SIZE == 5
