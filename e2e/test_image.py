import os
import subprocess
from datetime import datetime, timezone

import pytest
from pymongo import MongoClient

from tests.fixtures import assert_contract, cluster_for, event, log_lines, seed_articles

pytestmark = pytest.mark.skipif(
    not (os.environ.get("E2E_IMAGE") and os.environ.get("MONGODB_URI")),
    reason="runs in CI: needs E2E_IMAGE and MONGODB_URI",
)


def _run(image: str, uri: str) -> tuple[subprocess.CompletedProcess, list[dict]]:
    proc = subprocess.run(
        ["docker", "run", "--rm", "--network", "host", "-e", f"MONGODB_URI={uri}", "-e", "CLUSTERING_GPU=false", image],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc, log_lines(proc.stdout)


def test_image_clusters_seeded_articles_twice():
    uri, image = os.environ["MONGODB_URI"], os.environ["E2E_IMAGE"]
    db = MongoClient(uri).get_default_database()
    for name in ("news-article", "news-cluster", "cluster-article-link"):
        db[name].drop()
    seed = seed_articles(db, datetime.now(timezone.utc))

    _, lines = _run(image, uri)
    assert event(lines, "run_start")["mongoCompressor"] == "zstd"
    summary = event(lines, "run_summary")
    assert summary["windowArticles"] == 9
    assert summary["multiArticleClusters"] == 2
    assert_contract(db)
    cluster_a = {cluster_for(db, a)["_id"] for a in seed.group_a}
    cluster_b = {cluster_for(db, b)["_id"] for b in seed.group_b}
    assert len(cluster_a) == 1 and len(cluster_b) == 1 and cluster_a != cluster_b
    stable_ids = {c["stableClusterId"] for c in db["news-cluster"].find({"stableClusterId": {"$exists": True}})}
    assert len(stable_ids) == 2

    _, lines = _run(image, uri)
    second = event(lines, "run_summary")
    assert second["stableCarried"] == 2
    assert {c["stableClusterId"] for c in db["news-cluster"].find({"stableClusterId": {"$exists": True}})} == stable_ids
    run_ids = {d["clusteringRunId"] for coll in ("news-cluster", "cluster-article-link") for d in db[coll].find()}
    assert len(run_ids) == 1
    assert run_ids == {second["clusteringRunId"]}
