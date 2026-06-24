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

import numpy as np
import pytest

from dimos.memory2.blobstore.postgres import PostgresBlobStore
from dimos.memory2.notifier.subject import SubjectNotifier
from dimos.memory2.observationstore.postgres import PostgresObservationStore
from dimos.memory2.registry import qual
from dimos.memory2.store.postgres import PostgresRegistryStore, PostgresStore
from dimos.memory2.type.observation import _UNLOADED
from dimos.memory2.vectorstore.postgres import PostgresVectorStore
from dimos.models.embedding.base import Embedding

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


def test_postgres_scalar_stream_queries_and_reopens(postgres_conn: object) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        stream = store.stream("numbers", int)
        stream.append(1, ts=10.0, pose=(0, 0, 0), tags={"kind": "odd"})
        stream.append(2, ts=20.0, pose=(1, 0, 0), tags={"kind": "even"})
        stream.append(3, ts=30.0, pose=(5, 0, 0), tags={"kind": "odd"})

        point_row = postgres_conn.execute(  # type: ignore[attr-defined]
            """
            SELECT public.GeometryType(pose_point), public.ST_Z(pose_point)
            FROM numbers
            WHERE id = 1
            """
        ).fetchone()
        assert point_row == ("POINT", 0.0)
        assert stream.count() == 3
        assert stream.exists()
        assert stream.first().data == 1
        assert stream.last().data == 3
        assert type(stream.first().data) is int
        assert [o.data for o in stream.after(15.0).to_list()] == [2, 3]
        assert [o.data for o in stream.tags(kind="odd").to_list()] == [1, 3]
        assert [o.data for o in stream.near((0, 0, 0), radius=1.5).to_list()] == [1, 2]
        assert [o.data for o in stream.order_by("ts", desc=True).offset(1).limit(1)] == [2]

        floats = store.stream("floats", float)
        floats.append(1.0, ts=1.0)
        floats.append(1.5, ts=2.0)
        float_data = [obs.data for obs in floats.to_list()]
        assert [(value, type(value)) for value in float_data] == [(1, int), (1.5, float)]

    with PostgresStore(conn=postgres_conn) as reopened:
        stream = reopened.stream("numbers")
        assert [o.data for o in stream.to_list()] == [1, 2, 3]


def test_postgres_blob_store_lazy_and_eager_loading(postgres_conn: object) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        stream = store.stream("logs", str)
        stream.append("first", ts=1.0)
        stream.append("second", ts=2.0)

        assert postgres_conn.execute("SELECT to_regclass('logs_blob')").fetchone()[0] == "logs_blob"  # type: ignore[attr-defined]
        rows = stream.to_list()
        assert isinstance(rows[0]._data, type(_UNLOADED))
        assert rows[0]._loader is not None
        assert [o.data for o in rows] == ["first", "second"]

    with PostgresStore(conn=postgres_conn) as reopened:
        stream = reopened.stream("logs")
        assert reopened.stream("logs") is stream
        assert [o.data for o in stream.to_list()] == ["first", "second"]

    with PostgresStore(conn=postgres_conn) as store:
        eager = store.stream("eager_logs", str, eager_blobs=True)
        eager.append("payload", ts=1.0)
        rows = eager.to_list()
        assert rows[0].data == "payload"
        assert not isinstance(rows[0]._data, type(_UNLOADED))


def test_postgres_delete_stream_removes_metadata_blobs_and_registry(
    postgres_conn: object,
) -> None:
    with PostgresStore(conn=postgres_conn) as store:
        stream = store.stream("temporary", str)
        stream.append("payload", ts=1.0)
        store.delete_stream("temporary")

    assert PostgresRegistryStore(postgres_conn).get("temporary") is None
    table = postgres_conn.execute("SELECT to_regclass('temporary')").fetchone()[0]  # type: ignore[attr-defined]
    assert table is None
    blob_table = postgres_conn.execute("SELECT to_regclass('temporary_blob')").fetchone()[0]  # type: ignore[attr-defined]
    assert blob_table is None


def test_postgres_vector_search_with_pgvector(postgres_conn: object) -> None:
    def emb(vec: list[float]) -> Embedding:
        arr = np.array(vec, dtype=np.float32)
        return Embedding(vector=arr / (np.linalg.norm(arr) + 1e-10))

    with PostgresStore(conn=postgres_conn) as store:
        stream = store.stream("vecs", str)
        vector_store = stream._source.vector_store
        if not getattr(vector_store, "_available", False):
            pytest.skip("pgvector extension is unavailable")

        stream.append("north", ts=1.0, embedding=emb([0, 1, 0]))
        stream.append("east", ts=2.0, embedding=emb([1, 0, 0]))
        stream.append("south", ts=3.0, embedding=emb([0, -1, 0]))

        assert postgres_conn.execute("SELECT to_regclass('vecs_vec')").fetchone()[0] == "vecs_vec"  # type: ignore[attr-defined]
        results = stream.search(emb([0, 1, 0]), k=2).to_list()
        assert [o.data for o in results] == ["north", "east"]
        assert results[0].similarity is not None
        assert results[0].similarity > 0.99

        store.delete_stream("vecs")
        assert postgres_conn.execute("SELECT to_regclass('vecs_vec')").fetchone()[0] is None  # type: ignore[attr-defined]
