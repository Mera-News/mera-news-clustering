"""Shared pytest fixtures: settings and a structured-log sink."""
from __future__ import annotations

import pytest

from src.config import Settings


@pytest.fixture
def make_settings():
    def _make(**overrides) -> Settings:
        return Settings(**overrides)

    return _make


@pytest.fixture
def settings(make_settings) -> Settings:
    return make_settings()


# --------------------------------------------------------------------------- #
# Structured-log capture: a JobLogger sink that records parsed JSON lines.
# --------------------------------------------------------------------------- #
@pytest.fixture
def log_sink():
    import json

    lines: list[dict] = []

    def write(line: str) -> None:
        lines.append(json.loads(line))

    write.lines = lines  # type: ignore[attr-defined]
    return write


# --------------------------------------------------------------------------- #
# Real mongod for the integration tests (started lazily, one per session).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def mongod_base_uri(tmp_path_factory):
    import os
    import shutil
    import socket
    import subprocess
    import time

    import pymongo

    binary = os.environ.get("MONGOD_BIN") or shutil.which("mongod")
    if not binary or not os.access(binary, os.X_OK):
        pytest.fail("Integration tests need a MongoDB server binary (mongod 8.x): set MONGOD_BIN or put mongod on PATH.")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    dbpath = tmp_path_factory.mktemp("mongod")
    proc = subprocess.Popen(
        [binary, "--dbpath", str(dbpath), "--port", str(port), "--bind_ip", "127.0.0.1", "--quiet"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    base = f"mongodb://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with pymongo.MongoClient(base, serverSelectionTimeoutMS=500) as probe:
                    probe.admin.command("ping")
                break
            except pymongo.errors.PyMongoError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    pytest.fail("mongod did not become ready within 30 s")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@pytest.fixture
def mongo_uri(mongod_base_uri):
    import uuid

    import pymongo

    name = f"it_{uuid.uuid4().hex[:12]}"
    yield f"{mongod_base_uri}/{name}"
    with pymongo.MongoClient(mongod_base_uri) as client:
        client.drop_database(name)


@pytest.fixture
def clean_job_env(monkeypatch):
    from src.config import Settings
    from src.equivalence_main import EquivSettings

    names = {*Settings.model_fields, "CLOUD_RUN_JOB", "CLOUD_RUN_EXECUTION"}
    names |= {n for n in EquivSettings.model_fields if n.startswith("EQUIV_")}
    for name in names:
        monkeypatch.delenv(name, raising=False)
