"""The clustering job end to end against a real mongod."""
from datetime import datetime, timezone

import pymongo
import pytest

from src import job_main_mongo
from tests.fixtures import assert_contract, cluster_for, event, log_lines, seed_articles


@pytest.fixture
def db(mongo_uri):
    with pymongo.MongoClient(mongo_uri) as client:
        yield client.get_default_database()


@pytest.fixture
def run_job(clean_job_env, mongo_uri, monkeypatch, capsys):
    monkeypatch.setenv("MONGODB_URI", mongo_uri)

    def run(**env) -> list[dict]:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        capsys.readouterr()
        assert job_main_mongo.main() == 0
        return log_lines(capsys.readouterr().out)

    return run


def _run_id_set(db) -> set[str]:
    return {d["clusteringRunId"] for c in ("news-cluster", "cluster-article-link") for d in db[c].find()}


def test_first_run_writes_contract_shaped_clusters(db, run_job):
    seed = seed_articles(db, datetime.now(timezone.utc))
    lines = run_job()

    assert event(lines, "run_start")["mongoCompressor"] == "zstd"
    summary = event(lines, "run_summary")
    assert {k: summary[k] for k in (
        "windowArticles", "multiArticleClusters", "singletons", "maxClusterSize", "stableMinted", "stableCarried"
    )} == {
        "windowArticles": 9, "multiArticleClusters": 2, "singletons": 2,
        "maxClusterSize": 4, "stableMinted": 2, "stableCarried": 0,
    }
    a = {cluster_for(db, i)["_id"] for i in seed.group_a}
    b = {cluster_for(db, i)["_id"] for i in seed.group_b}
    assert len(a) == 1 and len(b) == 1 and a != b
    assert cluster_for(db, seed.group_a[0])["clusterSize"] == 4
    assert cluster_for(db, seed.group_b[0])["clusterSize"] == 3
    outlier_clusters = [cluster_for(db, i) for i in seed.outliers]
    assert [c["clusterSize"] for c in outlier_clusters] == [1, 1]
    assert outlier_clusters[0]["_id"] != outlier_clusters[1]["_id"]
    assert_contract(db)
    assert _run_id_set(db) == {summary["clusteringRunId"]}


def test_second_run_carries_stable_ids_and_swaps_generation(db, run_job):
    seed = seed_articles(db, datetime.now(timezone.utc))
    run_job()
    stable = {cluster_for(db, i)["stableClusterId"] for i in (seed.group_a[0], seed.group_b[0])}
    counts = (db["news-cluster"].count_documents({}), db["cluster-article-link"].count_documents({}))

    summary = event(run_job(), "run_summary")

    assert summary["stableCarried"] == 2
    assert {cluster_for(db, i)["stableClusterId"] for i in (seed.group_a[0], seed.group_b[0])} == stable
    assert _run_id_set(db) == {summary["clusteringRunId"]}
    assert (db["news-cluster"].count_documents({}), db["cluster-article-link"].count_documents({})) == counts
    assert_contract(db)


def test_singletons_can_be_left_unpersisted(db, run_job):
    seed = seed_articles(db, datetime.now(timezone.utc))
    run_job(PERSIST_SINGLETON_CLUSTERS="false")

    assert db["news-cluster"].count_documents({"clusterSize": 1}) == 0
    assert db["cluster-article-link"].count_documents({"articleId": {"$in": seed.outliers}}) == 0
    assert db["news-article"].count_documents(
        {"_id": {"$in": seed.window}, "processingStatus": "aggregation_successful"}
    ) == 9
    assert_contract(db)


def test_legacy_wipe_first_leaves_one_generation(db, run_job):
    seed_articles(db, datetime.now(timezone.utc))
    run_job(PERSIST_LEGACY_WIPE_FIRST="true")
    summary = event(run_job(), "run_summary")

    assert db["news-cluster"].count_documents({}) == 4
    assert db["cluster-article-link"].count_documents({}) == 9
    assert _run_id_set(db) == {summary["clusteringRunId"]}
    assert_contract(db)


def test_empty_window_is_a_clean_exit(db, run_job):
    summary = event(run_job(), "run_summary")

    assert summary["windowArticles"] == 0
    assert db["news-cluster"].count_documents({}) == 0
    assert db["cluster-article-link"].count_documents({}) == 0
