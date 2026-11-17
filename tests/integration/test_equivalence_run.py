"""The equivalence tool end to end against a real mongod."""
from datetime import datetime, timezone

import pymongo

from src import equivalence_main
from tests.fixtures import event, log_lines, seed_articles


def test_equivalence_fit_compare_drop(clean_job_env, mongo_uri, monkeypatch, capsys):
    now = datetime.now(timezone.utc)
    with pymongo.MongoClient(mongo_uri) as client:
        db = client.get_default_database()
        seed_articles(db, now)
        monkeypatch.setenv("MONGODB_URI", mongo_uri)
        monkeypatch.setenv("EQUIV_WINDOW_END", now.isoformat())
        monkeypatch.setenv("EQUIV_EPSILONS", "0")

        for tag in ("a", "b"):
            monkeypatch.setenv("EQUIV_TAG", tag)
            assert equivalence_main.main() == 0
        assert db["ml-equivalence-run"].count_documents({}) == 2

        monkeypatch.delenv("EQUIV_TAG")
        monkeypatch.setenv("EQUIV_COMPARE", "a,b")
        capsys.readouterr()
        assert equivalence_main.main() == 0
        assert event(log_lines(capsys.readouterr().out), "equivalence_verdict")["verdict"] == "IDENTICAL"

        monkeypatch.setenv("EQUIV_DROP", "true")
        assert equivalence_main.main() == 0
        assert db["ml-equivalence-run"].count_documents({}) == 0
