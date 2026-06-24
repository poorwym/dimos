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

from typing import Any

from pydantic import Field

from dimos.memory2.blobstore.base import BlobStore, BlobStoreConfig
from dimos.memory2.utils.validation import validate_identifier


class PostgresBlobStoreConfig(BlobStoreConfig):
    conn: Any = Field(exclude=True)


class PostgresBlobStore(BlobStore):
    """Stores blobs in a separate Postgres bytea table per stream."""

    config: PostgresBlobStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conn = self.config.conn
        self._tables: set[str] = set()

    def start(self) -> None:
        pass

    def _ensure_table(self, stream_name: str) -> None:
        if stream_name in self._tables:
            return
        validate_identifier(stream_name)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{stream_name}_blob" (
                id   bigint PRIMARY KEY,
                data bytea NOT NULL
            )
            """
        )
        self._tables.add(stream_name)

    def put(self, stream_name: str, key: int, data: bytes) -> None:
        validate_identifier(stream_name)
        self._ensure_table(stream_name)
        self._conn.execute(
            f"""
            INSERT INTO "{stream_name}_blob" (id, data)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
            """,
            (key, data),
        )

    def get(self, stream_name: str, key: int) -> bytes:
        validate_identifier(stream_name)
        try:
            row = self._conn.execute(
                f'SELECT data FROM "{stream_name}_blob" WHERE id = %s',
                (key,),
            ).fetchone()
        except Exception:
            rollback = getattr(self._conn, "rollback", None)
            if rollback is not None:
                rollback()
            raise KeyError(f"No blob for stream={stream_name!r}, key={key}") from None
        if row is None:
            raise KeyError(f"No blob for stream={stream_name!r}, key={key}")
        data = row["data"] if isinstance(row, dict) else row[0]
        return bytes(data)

    def delete(self, stream_name: str, key: int) -> None:
        validate_identifier(stream_name)
        try:
            cur = self._conn.execute(
                f'DELETE FROM "{stream_name}_blob" WHERE id = %s',
                (key,),
            )
        except Exception:
            rollback = getattr(self._conn, "rollback", None)
            if rollback is not None:
                rollback()
            raise KeyError(f"No blob for stream={stream_name!r}, key={key}") from None
        if cur.rowcount == 0:
            raise KeyError(f"No blob for stream={stream_name!r}, key={key}")
