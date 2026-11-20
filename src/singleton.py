"""Cloud Run singleton guard — at most one execution of this job at a time.

Cloud Run Jobs have no concurrencyPolicy: RunJob happily starts a second
execution while one is running, so overlapping runs would double the (GPU)
clustering work. The rule: list this job's executions; if any OTHER
still-running execution is OLDER (tie-broken by name), yield with exit 0.
The oldest execution always survives, so the check cannot deadlock.

Outside Cloud Run (no CLOUD_RUN_EXECUTION env) the guard is a no-op, and any
API failure means "run" — the guard is an optimization, never a gate.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from src.job_logger import JobLogger


@dataclass
class SingletonDecision:
    run: bool
    blocker: str | None = None


def _default_list_executions(project: str, region: str, job: str) -> list[dict]:
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    session = google.auth.transport.requests.AuthorizedSession(credentials)
    url = (
        f"https://run.googleapis.com/v2/projects/{project}"
        f"/locations/{region}/jobs/{job}/executions?pageSize=100"
    )
    resp = session.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.json().get("executions", [])


# Cancelled executions keep completionTime unset for tens of seconds while
# finalizing — one snapshot can see a zombie as "running". Re-check before
# yielding: zombies clear within seconds, real runs persist.
RECHECKS = 2
RECHECK_DELAY_S = 15.0


def singleton_decision(
    project: str,
    region: str,
    logger: JobLogger,
    list_executions=_default_list_executions,
    environ: dict[str, str] | None = None,
    sleep=time.sleep,
) -> SingletonDecision:
    env = environ if environ is not None else os.environ
    job = env.get("CLOUD_RUN_JOB", "")
    execution = env.get("CLOUD_RUN_EXECUTION", "")
    if not job or not execution or not project or not region:
        return SingletonDecision(run=True)

    decision = SingletonDecision(run=True)
    for attempt in range(RECHECKS + 1):
        if attempt > 0:
            sleep(RECHECK_DELAY_S)
        decision = _decide_once(project, region, job, execution, logger, list_executions)
        if decision.run:
            return decision
    return decision


def _decide_once(
    project: str,
    region: str,
    job: str,
    execution: str,
    logger: JobLogger,
    list_executions,
) -> SingletonDecision:
    try:
        executions = list_executions(project, region, job)
    except Exception as err:  # noqa: BLE001 - guard must never block real work
        logger.warn("singleton_check_failed", error=str(err))
        return SingletonDecision(run=True)

    # An execution cancelled while queued can keep completionTime unset
    # FOREVER (observed: cancelledCount=1, no completionTime) —
    # terminal task counts also mean "not running" (task_count is 1).
    def _is_running(e: dict) -> bool:
        if e.get("completionTime"):
            return False
        return not (
            e.get("succeededCount") or e.get("failedCount") or e.get("cancelledCount")
        )

    running = [e for e in executions if _is_running(e)]
    mine = next(
        (e for e in running if e.get("name", "").endswith(f"/{execution}")), None
    )
    if mine is None:  # API lag — run rather than risk a global skip
        return SingletonDecision(run=True)

    my_create = mine.get("createTime", "")
    my_name = mine.get("name", "")
    for other in running:
        if other is mine:
            continue
        o_create = other.get("createTime", "")
        o_name = other.get("name", "")
        if o_create < my_create or (o_create == my_create and o_name < my_name):
            return SingletonDecision(run=False, blocker=o_name)
    return SingletonDecision(run=True)
