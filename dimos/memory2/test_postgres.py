# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from dimos.memory2.blobstore.postgres import PostgresBlobStore
from dimos.memory2.notifier.subject import SubjectNotifier
from dimos.memory2.observationstore.postgres import PostgresObservationStore
from dimos.memory2.registry import qual
from dimos.memory2.store.postgres import PostgresRegistryStore, PostgresStore
from dimos.memory2.vectorstore.postgres import PostgresVectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator


def _postgres_dsn() -> str:
    """Connection settings from docker/postgres/docker-compose.yaml."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "dimos")
    user = os.environ.get("POSTGRES_USER", "dimos")
    password = os.environ.get("POSTGRES_PASSWORD", "dimos")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@pytest.fixture
def postgres_conn() -> Iterator[object]:
    psycopg = pytest.importorskip("psycopg")
    schema = f"memory2_test_{uuid4().hex}"

    try:
        conn = psycopg.connect(_postgres_dsn(), connect_timeout=2)
    except Exception as exc:
        pytest.skip(f"Postgres test database is unavailable: {exc}")

    try:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(f'SET search_path TO "{schema}"')
        conn.commit()
        yield conn
    finally:
        try:
            conn.execute("SET search_path TO public")
            conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            conn.commit()
        finally:
            conn.close()


def test_postgres_registry_round_trips_json_config(postgres_conn: object) -> None:
    registry = PostgresRegistryStore(postgres_conn)

    config = {"payload_module": "builtins.str", "codec_id": "pickle", "eager_blobs": True}
    registry.put("logs", config)

    assert registry.get("logs") == config
    assert registry.list_streams() == ["logs"]


def test_postgres_store_creates_stream_and_persists_registry_config(
    postgres_conn: object,
) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        stream = store.stream("logs", str, eager_blobs=True)
        backend = stream._source

    stored = PostgresRegistryStore(postgres_conn).get("logs")
    assert stored == {
        "payload_module": "builtins.str",
        "codec_id": "pickle",
        "eager_blobs": True,
        "blob_store": {"class": qual(PostgresBlobStore), "config": {}},
        "vector_store": {"class": qual(PostgresVectorStore), "config": {}},
        "notifier": {"class": qual(SubjectNotifier), "config": {}},
    }
    assert isinstance(backend.metadata_store, PostgresObservationStore)
    assert isinstance(backend.blob_store, PostgresBlobStore)
    assert isinstance(backend.vector_store, PostgresVectorStore)


def test_postgres_store_reopens_stream_without_payload_type(postgres_conn: object) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        store.stream("logs", str, eager_blobs=True)

    with PostgresStore(conn=postgres_conn) as reopened:
        stream = reopened.stream("logs")
        backend = stream._source

    assert backend.data_type is str
    assert backend.eager_blobs is True
    assert isinstance(backend.metadata_store, PostgresObservationStore)
    assert backend.metadata_store.name == "logs"
    assert sorted(reopened.list_streams()) == ["logs"]


def test_postgres_store_returns_cached_stream(postgres_conn: object) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        first = store.stream("logs", str)
        second = store.stream("logs", str)

    assert first is second


def test_postgres_store_rejects_type_mismatch_on_reopen(postgres_conn: object) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        store.stream("logs", str)

    with PostgresStore(conn=postgres_conn) as reopened:
        with pytest.raises(ValueError, match="was created with type"):
            reopened.stream("logs", int)


def test_postgres_store_rejects_invalid_stream_name(postgres_conn: object) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        with pytest.raises(ValueError, match="Invalid stream name"):
            store.stream("bad-name", str)


def test_postgres_component_serialization_excludes_live_connection(
    postgres_conn: object,
) -> None:
    assert PostgresBlobStore(conn=postgres_conn).serialize() == {
        "class": qual(PostgresBlobStore),
        "config": {},
    }
    assert PostgresVectorStore(conn=postgres_conn).serialize() == {
        "class": qual(PostgresVectorStore),
        "config": {},
    }
    assert PostgresObservationStore(conn=postgres_conn, name="logs").serialize() == {
        "class": qual(PostgresObservationStore),
        "config": {"name": "logs"},
    }
