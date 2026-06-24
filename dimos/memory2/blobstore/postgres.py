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


class PostgresBlobStoreConfig(BlobStoreConfig):
    conn: Any = Field(exclude=True)


class PostgresBlobStore(BlobStore):
    """Postgres bytea blob store shell.

    Storage SQL is implemented under the dedicated PostgresBlobStore issue.
    """

    config: PostgresBlobStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conn = self.config.conn

    def put(self, stream_name: str, key: int, data: bytes) -> None:
        raise NotImplementedError("PostgresBlobStore.put is not implemented yet")

    def get(self, stream_name: str, key: int) -> bytes:
        raise NotImplementedError("PostgresBlobStore.get is not implemented yet")

    def delete(self, stream_name: str, key: int) -> None:
        raise NotImplementedError("PostgresBlobStore.delete is not implemented yet")
