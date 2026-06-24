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

"""SQLite -> PostgreSQL migration for memory2 stores."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np

from dimos.memory2.blobstore.postgres import PostgresBlobStore
from dimos.memory2.notifier.subject import SubjectNotifier
from dimos.memory2.registry import qual
from dimos.memory2.store.postgres import PostgresRegistryStore, PostgresStore
from dimos.memory2.utils.sqlite import open_sqlite_connection
from dimos.memory2.utils.validation import validate_identifier
from dimos.memory2.vectorstore.postgres import PostgresVectorStore


@dataclass(frozen=True)
class MigrationResult:
    streams: int = 0
    observations: int = 0
    blobs: int = 0
    vectors: int = 0


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sqlite_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'virtual table') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _postgres_table_exists(conn: Any, name: str) -> bool:
    row = conn.execute("SELECT to_regclass(%s)", (name,)).fetchone()
    if row is None:
        return False
    value = row["to_regclass"] if isinstance(row, dict) else row[0]
    return value is not None


def _sqlite_stream_configs(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    if not _sqlite_table_exists(conn, "_streams"):
        return {}
    rows = conn.execute("SELECT name, config FROM _streams ORDER BY name").fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


def _target_config(source: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "payload_module": source["payload_module"],
        "codec_id": source["codec_id"],
        "eager_blobs": source.get("eager_blobs", False),
        "blob_store": {"class": qual(PostgresBlobStore), "config": {}},
        "vector_store": {"class": qual(PostgresVectorStore), "config": {}},
        "notifier": source.get(
            "notifier",
            {"class": qual(SubjectNotifier), "config": {}},
        ),
    }
    return cfg


def _ensure_postgres_metadata_table(conn: Any, name: str) -> None:
    qname = _quote_ident(name)
    conn.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {qname} (
            id      bigserial PRIMARY KEY,
            ts      double precision NOT NULL,
            value   numeric,
            pose_x  double precision,
            pose_y  double precision,
            pose_z  double precision,
            pose_qx double precision,
            pose_qy double precision,
            pose_qz double precision,
            pose_qw double precision,
            pose_point public.geometry(PointZ, 0),
            tags    jsonb NOT NULL DEFAULT '{{}}'::jsonb
        )
        """
    )
    conn.execute(f"ALTER TABLE {qname} ADD COLUMN IF NOT EXISTS pose_point public.geometry(PointZ, 0)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {_quote_ident(name + '_ts_idx')} ON {qname} (ts)")
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_quote_ident(name + '_tags_idx')} "
        f"ON {qname} USING gin (tags)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {_quote_ident(name + '_pose_point_idx')} "
        f"ON {qname} USING gist (pose_point public.gist_geometry_ops_nd) "
        "WHERE pose_point IS NOT NULL"
    )


def _ensure_postgres_blob_table(conn: Any, stream: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote_ident(stream + '_blob')} (
            id   bigint PRIMARY KEY,
            data bytea NOT NULL
        )
        """
    )


def _ensure_postgres_vector_table(conn: Any, stream: str, dim: int) -> None:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_quote_ident(stream + '_vec')} (
            id        bigint PRIMARY KEY,
            embedding public.vector({dim}) NOT NULL
        )
        """
    )


def _iter_batches(rows: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _sqlite_tags_json(raw: Any) -> str:
    if raw is None:
        return "{}"
    if isinstance(raw, str):
        return raw
    return json.dumps(dict(raw))


def _migrate_metadata(
    sqlite_conn: sqlite3.Connection,
    pg_conn: Any,
    stream: str,
    *,
    batch_size: int,
) -> int:
    if not _sqlite_table_exists(sqlite_conn, stream):
        return 0

    _ensure_postgres_metadata_table(pg_conn, stream)
    rows = sqlite_conn.execute(
        f"""
        SELECT
            id, ts, value,
            pose_x, pose_y, pose_z,
            pose_qx, pose_qy, pose_qz, pose_qw,
            json(tags)
        FROM {_quote_ident(stream)}
        ORDER BY id
        """
    )
    insert_sql = f"""
        INSERT INTO {_quote_ident(stream)} (
            id, ts, value,
            pose_x, pose_y, pose_z,
            pose_qx, pose_qy, pose_qz, pose_qw,
            pose_point,
            tags
        )
        VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            CASE
                WHEN %s THEN public.ST_SetSRID(
                    public.ST_MakePoint(%s, %s, %s),
                    0
                )::public.geometry(PointZ, 0)
                ELSE NULL
            END,
            %s::jsonb
        )
        ON CONFLICT (id) DO UPDATE SET
            ts = EXCLUDED.ts,
            value = EXCLUDED.value,
            pose_x = EXCLUDED.pose_x,
            pose_y = EXCLUDED.pose_y,
            pose_z = EXCLUDED.pose_z,
            pose_qx = EXCLUDED.pose_qx,
            pose_qy = EXCLUDED.pose_qy,
            pose_qz = EXCLUDED.pose_qz,
            pose_qw = EXCLUDED.pose_qw,
            pose_point = EXCLUDED.pose_point,
            tags = EXCLUDED.tags
    """

    total = 0
    max_id = 0
    with pg_conn.cursor() as cur:
        for batch in _iter_batches(rows, batch_size):
            params = []
            for row in batch:
                row_id, ts, value, px, py, pz, qx, qy, qz, qw, tags_json = row
                has_pose = px is not None
                max_id = max(max_id, int(row_id))
                params.append(
                    (
                        row_id,
                        ts,
                        value,
                        px,
                        py,
                        pz,
                        qx,
                        qy,
                        qz,
                        qw,
                        has_pose,
                        px,
                        py,
                        pz,
                        _sqlite_tags_json(tags_json),
                    )
                )
            cur.executemany(insert_sql, params)
            total += len(params)

    if max_id:
        pg_conn.execute(
            "SELECT setval(pg_get_serial_sequence(%s, 'id'), %s, true)",
            (stream, max_id),
        )
    return total


def _migrate_blobs(
    sqlite_conn: sqlite3.Connection,
    pg_conn: Any,
    stream: str,
    *,
    batch_size: int,
) -> int:
    source_table = f"{stream}_blob"
    if not _sqlite_table_exists(sqlite_conn, source_table):
        return 0

    _ensure_postgres_blob_table(pg_conn, stream)
    rows = sqlite_conn.execute(
        f"SELECT id, data FROM {_quote_ident(source_table)} ORDER BY id"
    )
    insert_sql = (
        f"INSERT INTO {_quote_ident(source_table)} (id, data) VALUES (%s, %s) "
        "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data"
    )
    total = 0
    with pg_conn.cursor() as cur:
        for batch in _iter_batches(rows, batch_size):
            cur.executemany(insert_sql, [(row[0], bytes(row[1])) for row in batch])
            total += len(batch)
    return total


def _decode_sqlite_vec(value: Any) -> list[float]:
    if isinstance(value, bytes | bytearray | memoryview):
        return np.frombuffer(value, dtype=np.float32).astype(float).tolist()
    if isinstance(value, str):
        return [float(v) for v in json.loads(value)]
    return [float(v) for v in value]


def _migrate_vectors(
    sqlite_conn: sqlite3.Connection,
    pg_conn: Any,
    stream: str,
    *,
    batch_size: int,
) -> int:
    source_table = f"{stream}_vec"
    if not _sqlite_table_exists(sqlite_conn, source_table):
        return 0

    rows = sqlite_conn.execute(
        f"SELECT rowid, embedding FROM {_quote_ident(source_table)} ORDER BY rowid"
    ).fetchall()
    if not rows:
        return 0

    first_vec = _decode_sqlite_vec(rows[0][1])
    dim = len(first_vec)
    _ensure_postgres_vector_table(pg_conn, stream, dim)
    insert_sql = (
        f"INSERT INTO {_quote_ident(source_table)} (id, embedding) "
        f"VALUES (%s, %s::public.vector({dim})) "
        "ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding"
    )

    total = 0
    with pg_conn.cursor() as cur:
        for batch in _iter_batches(rows, batch_size):
            params = []
            for row_id, raw_vec in batch:
                vec = _decode_sqlite_vec(raw_vec)
                if len(vec) != dim:
                    raise ValueError(
                        f"Vector dimension mismatch in {source_table}: expected {dim}, got {len(vec)}"
                    )
                params.append((row_id, "[" + ",".join(str(v) for v in vec) + "]"))
            cur.executemany(insert_sql, params)
            total += len(params)
    return total


def migrate_sqlite_to_postgres(
    sqlite_path: str | Path,
    postgres_dsn: str,
    *,
    streams: list[str] | None = None,
    replace: bool = False,
    batch_size: int = 1000,
) -> MigrationResult:
    """Migrate a memory2 SQLite store to a PostgreSQL database."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    try:
        import psycopg
    except ImportError as exc:
        raise ImportError("Install psycopg to migrate to PostgreSQL") from exc

    sqlite_conn = open_sqlite_connection(sqlite_path)
    pg_conn = psycopg.connect(postgres_dsn)
    try:
        registry = PostgresRegistryStore(pg_conn)
        configs = _sqlite_stream_configs(sqlite_conn)
        selected = sorted(streams if streams is not None else configs.keys())
        unknown = sorted(set(selected) - set(configs))
        if unknown:
            raise ValueError(f"Unknown SQLite stream(s): {', '.join(unknown)}")

        result = MigrationResult()
        for stream in selected:
            validate_identifier(stream)
            if replace:
                PostgresStore(conn=pg_conn).delete_stream(stream)
            elif registry.get(stream) is not None or _postgres_table_exists(pg_conn, stream):
                raise ValueError(
                    f"Target stream {stream!r} already exists; pass replace=True/--replace"
                )

            observations = _migrate_metadata(
                sqlite_conn,
                pg_conn,
                stream,
                batch_size=batch_size,
            )
            blobs = _migrate_blobs(sqlite_conn, pg_conn, stream, batch_size=batch_size)
            vectors = _migrate_vectors(sqlite_conn, pg_conn, stream, batch_size=batch_size)
            registry.put(stream, _target_config(configs[stream]))
            result = MigrationResult(
                streams=result.streams + 1,
                observations=result.observations + observations,
                blobs=result.blobs + blobs,
                vectors=result.vectors + vectors,
            )

        pg_conn.commit()
        return result
    except BaseException:
        pg_conn.rollback()
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()
