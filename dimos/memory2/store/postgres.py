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

import json
from typing import Any

from pydantic import Field

from dimos.memory2.backend import Backend
from dimos.memory2.blobstore.base import BlobStore
from dimos.memory2.blobstore.postgres import PostgresBlobStore
from dimos.memory2.codecs.base import codec_id
from dimos.memory2.observationstore.postgres import PostgresObservationStore
from dimos.memory2.registry import deserialize_component, qual
from dimos.memory2.store.base import Store, StoreConfig
from dimos.memory2.utils.validation import validate_identifier
from dimos.memory2.vectorstore.base import VectorStore
from dimos.memory2.vectorstore.postgres import PostgresVectorStore


class PostgresRegistryStore:
    """Postgres persistence for stream name -> config JSONB."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS _streams ("
            "    name   text PRIMARY KEY,"
            "    config jsonb NOT NULL"
            ")"
        )
        self._commit()

    def _commit(self) -> None:
        commit = getattr(self._conn, "commit", None)
        if commit is not None:
            commit()

    @staticmethod
    def _decode_config(raw: Any) -> dict[str, Any]:
        if isinstance(raw, str):
            return json.loads(raw)  # type: ignore[no-any-return]
        return raw

    @staticmethod
    def _first(row: Any) -> Any:
        if isinstance(row, dict):
            return row["config"]
        return row[0]

    def get(self, name: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT config FROM _streams WHERE name = %s",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return self._decode_config(self._first(row))

    def put(self, name: str, config: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO _streams (name, config)
            VALUES (%s, %s::jsonb)
            ON CONFLICT (name) DO UPDATE SET config = EXCLUDED.config
            """,
            (name, json.dumps(config)),
        )
        self._commit()

    def delete(self, name: str) -> None:
        self._conn.execute("DELETE FROM _streams WHERE name = %s", (name,))
        self._commit()

    def list_streams(self) -> list[str]:
        rows = self._conn.execute("SELECT name FROM _streams").fetchall()
        return [row["name"] if isinstance(row, dict) else row[0] for row in rows]


class PostgresStoreConfig(StoreConfig):
    """Config for Postgres-backed memory2 stores."""

    dsn: str | None = None
    conn: Any = Field(default=None, exclude=True)


class PostgresStore(Store):
    """Store backed by a Postgres database."""

    config: PostgresStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._owns_conn = self.config.conn is None
        self._registry_conn = self.config.conn if self.config.conn is not None else self._connect()
        self._registry = PostgresRegistryStore(conn=self._registry_conn)

    def _connect(self) -> Any:
        if self.config.dsn is None:
            raise ValueError("PostgresStore requires either conn= or dsn=")
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError("Install psycopg to create PostgresStore from dsn=") from exc
        return psycopg.connect(self.config.dsn)

    def _component_conn(self) -> Any:
        return self._registry_conn

    def _assemble_backend(self, name: str, stored: dict[str, Any]) -> Backend[Any]:
        """Reconstruct a Backend from a stored config dict."""
        from dimos.memory2.codecs.base import _resolve_payload_type, codec_from_id
        from dimos.memory2.notifier.subject import SubjectNotifier

        payload_module = stored["payload_module"]
        codec = codec_from_id(stored["codec_id"], payload_module)
        data_type = _resolve_payload_type(payload_module)
        eager_blobs = stored.get("eager_blobs", False)
        conn = self._component_conn()

        bs_data = stored.get("blob_store")
        if bs_data is not None:
            if bs_data["class"] == qual(PostgresBlobStore):
                bs: Any = PostgresBlobStore(conn=conn)
            else:
                bs = deserialize_component(bs_data)
        else:
            bs = PostgresBlobStore(conn=conn)

        vs_data = stored.get("vector_store")
        if vs_data is not None:
            if vs_data["class"] == qual(PostgresVectorStore):
                vs: Any = PostgresVectorStore(conn=conn)
            else:
                vs = deserialize_component(vs_data)
        else:
            vs = PostgresVectorStore(conn=conn)

        notifier_data = stored.get("notifier")
        notifier = deserialize_component(notifier_data) if notifier_data else SubjectNotifier()

        return Backend(
            metadata_store=PostgresObservationStore(conn=conn, name=name),
            codec=codec,
            data_type=data_type,
            blob_store=bs,
            vector_store=vs,
            notifier=notifier,
            eager_blobs=eager_blobs,
        )

    @staticmethod
    def _serialize_backend(backend: Backend[Any], payload_module: str) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "payload_module": payload_module,
            "codec_id": codec_id(backend.codec),
            "eager_blobs": backend.eager_blobs,
        }
        if backend.blob_store is not None:
            cfg["blob_store"] = backend.blob_store.serialize()
        if backend.vector_store is not None:
            cfg["vector_store"] = backend.vector_store.serialize()
        cfg["notifier"] = backend.notifier.serialize()
        return cfg

    def _create_backend(
        self, name: str, payload_type: type[Any] | None = None, **config: Any
    ) -> Backend[Any]:
        validate_identifier(name)

        stored = self._registry.get(name)
        if stored is not None:
            if payload_type is not None:
                actual_module = f"{payload_type.__module__}.{payload_type.__qualname__}"
                if actual_module != stored["payload_module"]:
                    raise ValueError(
                        f"Stream {name!r} was created with type {stored['payload_module']}, "
                        f"but opened with {actual_module}"
                    )
            return self._assemble_backend(name, stored)

        if payload_type is None:
            raise TypeError(f"Stream {name!r} does not exist yet — payload_type is required")

        conn = self._component_conn()
        if not isinstance(config.get("blob_store"), BlobStore):
            config["blob_store"] = PostgresBlobStore(conn=conn)
        if not isinstance(config.get("vector_store"), VectorStore):
            config["vector_store"] = PostgresVectorStore(conn=conn)

        codec = self._resolve_codec(payload_type, config.get("codec"))
        config["codec"] = codec
        config["observation_store"] = PostgresObservationStore(conn=conn, name=name)

        backend = super()._create_backend(name, payload_type, **config)
        payload_module = f"{payload_type.__module__}.{payload_type.__qualname__}"
        self._registry.put(name, self._serialize_backend(backend, payload_module))
        return backend

    def list_streams(self) -> list[str]:
        db_names = set(self._registry.list_streams())
        return sorted(db_names | set(self._streams.keys()))

    def delete_stream(self, name: str) -> None:
        validate_identifier(name)
        super().delete_stream(name)
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}_blob"')
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}_vec"')
        self._registry_conn.execute(f'DROP TABLE IF EXISTS "{name}_rtree"')
        self._registry.delete(name)

    def stop(self) -> None:
        super().stop()
        if self._owns_conn:
            close = getattr(self._registry_conn, "close", None)
            if close is not None:
                close()
