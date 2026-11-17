"""Seed data and contract assertions shared by the integration and image tests."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from bson import ObjectId

CONTRACT: dict = json.loads((Path(__file__).parent / "contract" / "collections.json").read_text())
DIMS = 768

_PY_TYPES = {
    "objectId": ObjectId,
    "string": str,
    "date": datetime,
    "int": int,
    "double": float,
    "array": list,
}


@dataclass(frozen=True)
class Seed:
    group_a: list[ObjectId]  # 4 articles, one story
    group_b: list[ObjectId]  # 3 articles, another story
    outliers: list[ObjectId]  # 2 articles, unrelated
    stale_duplicate: ObjectId  # older copy of group_a[0]'s URL: deduped away
    too_old: ObjectId  # created 49 h ago: outside the 48 h window
    empty_vector: ObjectId  # vector []: excluded by the window query

    @property
    def window(self) -> list[ObjectId]:
        return self.group_a + self.group_b + self.outliers


def _type_ok(value, name: str) -> bool:
    if name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _PY_TYPES[name])


def seed_articles(db, now: datetime) -> Seed:
    rng = np.random.default_rng(7)
    counter = iter(range(1, 100))

    def vector(axis: int) -> list[float]:
        unit = np.zeros(DIMS)
        unit[axis] = 1.0
        return (unit + rng.normal(0, 1e-3, DIMS)).tolist()

    def insert(slug: str, hours_ago: int, vec: list[float], url: str | None = None) -> ObjectId:
        doc = {
            "article_url": url or f"https://news{next(counter)}.example.test/{slug}",
            "createdAt": now - timedelta(hours=hours_ago),
            "embedding_article_data_text_matching": vec,
            "processingStatus": "embedding_successful",
        }
        return db["news-article"].insert_one(doc).inserted_id

    group_a = [insert(f"story-a-{i}", 1 + i, vector(0)) for i in range(4)]
    group_b = [insert(f"story-b-{i}", 1 + i, vector(1)) for i in range(3)]
    outliers = [insert("outlier-c", 5, vector(2)), insert("outlier-d", 6, vector(3))]
    first_url = db["news-article"].find_one({"_id": group_a[0]})["article_url"]
    return Seed(
        group_a=group_a,
        group_b=group_b,
        outliers=outliers,
        stale_duplicate=insert("story-a-stale", 20, vector(0), url=first_url),
        too_old=insert("too-old", 49, vector(0)),
        empty_vector=insert("empty-vector", 2, []),
    )


def assert_contract(db) -> None:
    clusters = list(db["news-cluster"].find())
    links = list(db["cluster-article-link"].find())
    for coll, docs in (("news-cluster", clusters), ("cluster-article-link", links)):
        spec = CONTRACT[coll]
        allowed = {**spec["required"], **spec["optional"]}
        for doc in docs:
            assert set(spec["required"]) <= set(doc), f"{coll} missing {set(spec['required']) - set(doc)}"
            assert set(doc) <= set(allowed), f"{coll} has extra keys {set(doc) - set(allowed)}"
            for key, value in doc.items():
                assert _type_ok(value, allowed[key]), f"{coll}.{key} is {type(value).__name__}, want {allowed[key]}"
    for cluster in clusters:
        assert ("stableClusterId" in cluster) == (cluster["clusterSize"] >= 2), cluster
    by_id = {c["_id"]: c for c in clusters}
    for link in links:
        cluster = by_id.get(link["clusterId"])
        assert cluster is not None, f"link points at missing cluster {link['clusterId']}"
        assert link["clusteringRunId"] == cluster["clusteringRunId"], link

    stamp = CONTRACT["news-article"]["stamp"]
    stamped = db["news-article"].find({"processingStatus": CONTRACT["news-article"]["stampValues"]["processingStatus"]})
    window = {a["_id"] for a in stamped}
    for article in db["news-article"].find({"_id": {"$in": list(window)}}):
        for key, name in stamp.items():
            assert _type_ok(article.get(key), name), f"article {article['_id']} {key}"
    for article in db["news-article"].find({"_id": {"$nin": list(window)}}):
        assert article["processingStatus"] == "embedding_successful", article["_id"]
        assert "aggregationCompletedAt" not in article, article["_id"]


def cluster_for(db, article_id) -> dict:
    link = db["cluster-article-link"].find_one({"articleId": article_id})
    assert link is not None, f"no link for article {article_id}"
    return db["news-cluster"].find_one({"_id": link["clusterId"]})


def log_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def event(lines: list[dict], message: str) -> dict:
    for line in reversed(lines):
        if line.get("message") == message:
            return line
    raise AssertionError(f"no log line with message {message!r}")
