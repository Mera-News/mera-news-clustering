"""Singleton guard: one execution per Cloud Run job."""

from __future__ import annotations

import pytest

from src.job_logger import JobLogger
from src.singleton import singleton_decision


@pytest.fixture()
def logger() -> JobLogger:
    return JobLogger("test-job", level="error")


def _exec(exec_id: str, create: str, completion: str | None = None) -> dict:
    e = {
        "name": (
            "projects/p/locations/europe-west1/jobs/"
            f"test-job/executions/{exec_id}"
        ),
        "createTime": create,
    }
    if completion:
        e["completionTime"] = completion
    return e


ENV = {
    "CLOUD_RUN_JOB": "test-job",
    "CLOUD_RUN_EXECUTION": "test-job-mine1",
}


def test_runs_outside_cloud_run(logger):
    d = singleton_decision("p", "r", logger, list_executions=None, environ={})
    assert d.run is True


def test_yields_to_older_running_execution(logger):
    execs = [
        _exec("test-job-older1", "2026-07-07T16:54:56Z"),
        _exec("test-job-mine1", "2026-07-07T16:55:20Z"),
    ]
    d = singleton_decision(
        "p", "r", logger, list_executions=lambda *a: execs, environ=ENV, sleep=lambda s: None
    )
    assert d.run is False
    assert "older1" in d.blocker


def test_oldest_execution_survives(logger):
    execs = [
        _exec("test-job-mine1", "2026-07-07T16:55:20Z"),
        _exec("test-job-later", "2026-07-07T16:56:00Z"),
    ]
    d = singleton_decision(
        "p", "r", logger, list_executions=lambda *a: execs, environ=ENV, sleep=lambda s: None
    )
    assert d.run is True


def test_ignores_completed_executions(logger):
    execs = [
        _exec("test-job-done1", "2026-07-07T16:00:00Z", "2026-07-07T16:10:00Z"),
        _exec("test-job-mine1", "2026-07-07T16:55:20Z"),
    ]
    d = singleton_decision(
        "p", "r", logger, list_executions=lambda *a: execs, environ=ENV, sleep=lambda s: None
    )
    assert d.run is True


def test_tiebreak_equal_create_time_by_name(logger):
    t = "2026-07-07T16:55:20Z"
    execs = [
        _exec("test-job-aaaaa", t),
        _exec("test-job-mine1", t),
    ]
    d = singleton_decision(
        "p", "r", logger, list_executions=lambda *a: execs, environ=ENV, sleep=lambda s: None
    )
    assert d.run is False
    assert "aaaaa" in d.blocker


def test_runs_when_own_execution_not_listed(logger):
    execs = [_exec("test-job-other", "2026-07-07T16:54:00Z")]
    env = dict(ENV, CLOUD_RUN_EXECUTION="test-job-notlisted")
    d = singleton_decision(
        "p", "r", logger, list_executions=lambda *a: execs, environ=env, sleep=lambda s: None
    )
    assert d.run is True


def test_api_failure_never_blocks_work(logger):
    def boom(*a):
        raise RuntimeError("403")

    d = singleton_decision("p", "r", logger, list_executions=boom, environ=ENV, sleep=lambda s: None)
    assert d.run is True


def test_recheck_clears_cancellation_zombie(logger):
    zombie = _exec("test-job-zombie1", "2026-07-07T16:54:00Z")
    me = _exec("test-job-mine1", "2026-07-07T16:55:20Z")
    snapshots = iter([[zombie, me], [me]])
    calls = []
    d = singleton_decision(
        "p",
        "r",
        logger,
        list_executions=lambda *a: next(snapshots),
        environ=ENV,
        sleep=lambda s: calls.append(s),
    )
    assert d.run is True
    assert len(calls) == 1


def test_yields_only_after_blocker_persists(logger):
    older = _exec("test-job-older1", "2026-07-07T16:54:56Z")
    me = _exec("test-job-mine1", "2026-07-07T16:55:20Z")
    count = {"n": 0}

    def listing(*a):
        count["n"] += 1
        return [older, me]

    d = singleton_decision(
        "p", "r", logger, list_executions=listing, environ=ENV, sleep=lambda s: None
    )
    assert d.run is False
    assert count["n"] == 3


def test_ignores_cancelled_while_queued_zombie(logger):
    zombie = dict(
        _exec("test-job-cancel1", "2026-07-07T17:38:45Z"),
        cancelledCount=1,  # terminal, but Cloud Run never set completionTime
    )
    me = _exec("test-job-mine1", "2026-07-07T16:55:20Z")
    d = singleton_decision(
        "p",
        "r",
        logger,
        list_executions=lambda *a: [zombie, me],
        environ=ENV,
        sleep=lambda s: None,
    )
    assert d.run is True
