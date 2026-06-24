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

from dimos.memory2.vectorstore.base import VectorStore, VectorStoreConfig

if TYPE_CHECKING:
    from dimos.models.embedding.base import Embedding


class PostgresVectorStoreConfig(VectorStoreConfig):
    conn: Any = Field(exclude=True)


class PostgresVectorStore(VectorStore):
    """Postgres pgvector store shell.

    Vector SQL is implemented under the dedicated PostgresVectorStore issue.
    """

    config: PostgresVectorStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conn = self.config.conn

    def put(self, stream_name: str, key: int, embedding: Embedding) -> None:
        raise NotImplementedError("PostgresVectorStore.put is not implemented yet")

    def search(self, stream_name: str, query: Embedding, k: int | None) -> list[tuple[int, float]]:
        raise NotImplementedError("PostgresVectorStore.search is not implemented yet")

    def delete(self, stream_name: str, key: int) -> None:
        raise NotImplementedError("PostgresVectorStore.delete is not implemented yet")
