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

from typing import TYPE_CHECKING, Any

from pydantic import Field

from dimos.memory2.registry import qual
from dimos.memory2.utils.validation import validate_identifier
from dimos.memory2.vectorstore.base import VectorStore, VectorStoreConfig

if TYPE_CHECKING:
    from dimos.models.embedding.base import Embedding


class PostgresVectorStoreConfig(VectorStoreConfig):
    conn: Any = Field(exclude=True)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64


class PostgresVectorStore(VectorStore):
    """Vector store backed by Postgres and pgvector."""

    config: PostgresVectorStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conn = self.config.conn
        self._available = False
        self._unavailable_reason: Exception | None = None
        self._vector_type = "public.vector"
        self._distance_operator = "OPERATOR(public.<=>)"
        self._operator_class = "public.vector_cosine_ops"
        self._tables: dict[str, int] = {}

    @staticmethod
    def _vector_literal(embedding: Embedding) -> str:
        vec = embedding.to_numpy().astype(float).tolist()
        return "[" + ",".join(str(float(v)) for v in vec) + "]"

    def start(self) -> None:
        try:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            commit = getattr(self._conn, "commit", None)
            if commit is not None:
                commit()
            self._available = True
        except Exception as exc:
            rollback = getattr(self._conn, "rollback", None)
            if rollback is not None:
                rollback()
            self._available = False
            self._unavailable_reason = exc

    def _ensure_table(self, stream_name: str, dim: int) -> None:
        if stream_name in self._tables:
            return
        validate_identifier(stream_name)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{stream_name}_vec" (
                id        bigint PRIMARY KEY,
                embedding {self._vector_type}({dim}) NOT NULL
            )
            """
        )
        self._tables[stream_name] = dim

    def _require_available(self) -> None:
        if not self._available:
            raise RuntimeError(
                "PostgresVectorStore requires the pgvector extension to be installed"
            )

    def put(self, stream_name: str, key: int, embedding: Embedding) -> None:
        self._require_available()
        validate_identifier(stream_name)
        vec = self._vector_literal(embedding)
        dim = len(embedding.to_numpy())
        self._ensure_table(stream_name, dim)
        self._conn.execute(
            f"""
            INSERT INTO "{stream_name}_vec" (id, embedding)
            VALUES (%s, %s::{self._vector_type}({dim}))
            ON CONFLICT (id) DO UPDATE
            SET embedding = EXCLUDED.embedding
            """,
            (key, vec),
        )

    def put_many(self, stream_name: str, items: list[tuple[int, Embedding]]) -> None:
        if not items:
            return
        self._require_available()
        validate_identifier(stream_name)
        dim = len(items[0][1].to_numpy())
        self._ensure_table(stream_name, dim)
        params = []
        for key, embedding in items:
            if len(embedding.to_numpy()) != dim:
                raise ValueError("All embeddings in put_many must have the same dimensionality")
            params.append((key, self._vector_literal(embedding)))
        self._conn.cursor().executemany(
            f"""
            INSERT INTO "{stream_name}_vec" (id, embedding)
            VALUES (%s, %s::{self._vector_type}({dim}))
            ON CONFLICT (id) DO UPDATE
            SET embedding = EXCLUDED.embedding
            """,
            params,
        )

    def optimize(self, stream_name: str) -> None:
        self._require_available()
        validate_identifier(stream_name)
        self._conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS "{stream_name}_vec_embedding_hnsw_idx"
            ON "{stream_name}_vec"
            USING hnsw (embedding {self._operator_class})
            WITH (
                m = {self.config.hnsw_m},
                ef_construction = {self.config.hnsw_ef_construction}
            )
            """
        )

    def serialize(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        if self.config.hnsw_m != 16:
            cfg["hnsw_m"] = self.config.hnsw_m
        if self.config.hnsw_ef_construction != 64:
            cfg["hnsw_ef_construction"] = self.config.hnsw_ef_construction
        return {"class": qual(type(self)), "config": cfg}

    _DEFAULT_K = 4096

    def search(self, stream_name: str, query: Embedding, k: int | None) -> list[tuple[int, float]]:
        self._require_available()
        validate_identifier(stream_name)
        vec = self._vector_literal(query)
        dim = len(query.to_numpy())
        sql = (
            f"SELECT id, 1.0 - (embedding {self._distance_operator} "
            f"%s::{self._vector_type}({dim})) "
            "AS similarity "
            f'FROM "{stream_name}_vec" '
            f"ORDER BY embedding {self._distance_operator} "
            f"%s::{self._vector_type}({dim}) ASC, id ASC "
            "LIMIT %s"
        )
        params = [vec, vec, k if k is not None else self._DEFAULT_K]
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except Exception as exc:
            if "does not exist" in str(exc):
                rollback = getattr(self._conn, "rollback", None)
                if rollback is not None:
                    rollback()
                return []
            raise
        result: list[tuple[int, float]] = []
        for row in rows:
            obs_id = row["id"] if isinstance(row, dict) else row[0]
            similarity = row["similarity"] if isinstance(row, dict) else row[1]
            result.append((int(obs_id), max(0.0, float(similarity))))
        return result

    def delete(self, stream_name: str, key: int) -> None:
        if not self._available:
            return
        validate_identifier(stream_name)
        try:
            self._conn.execute(
                f'DELETE FROM "{stream_name}_vec" WHERE id = %s',
                (key,),
            )
        except Exception as exc:
            if "does not exist" not in str(exc):
                raise
            rollback = getattr(self._conn, "rollback", None)
            if rollback is not None:
                rollback()
