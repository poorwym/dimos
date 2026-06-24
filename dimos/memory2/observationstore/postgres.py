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

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import Field

from dimos.memory2.observationstore.base import ObservationStore, ObservationStoreConfig
from dimos.memory2.utils.validation import validate_identifier

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dimos.memory2.type.filter import StreamQuery
    from dimos.memory2.type.observation import Observation

T = TypeVar("T")


class PostgresObservationStoreConfig(ObservationStoreConfig):
    conn: Any = Field(exclude=True)
    name: str


class PostgresObservationStore(ObservationStore[T]):
    """Postgres metadata store shell.

    YIMO-72 only wires registry and backend assembly. Query/insert SQL belongs
    to the dedicated PostgresObservationStore implementation issue.
    """

    config: PostgresObservationStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        validate_identifier(self.config.name)
        self._conn = self.config.conn
        self._name = self.config.name

    @property
    def name(self) -> str:
        return self._name

    def insert(self, obs: Observation[T]) -> int:
        raise NotImplementedError("PostgresObservationStore.insert is not implemented yet")

    def query(self, q: StreamQuery) -> Iterator[Observation[T]]:
        raise NotImplementedError("PostgresObservationStore.query is not implemented yet")

    def count(self, q: StreamQuery) -> int:
        raise NotImplementedError("PostgresObservationStore.count is not implemented yet")

    def fetch_by_ids(self, ids: list[int]) -> list[Observation[T]]:
        raise NotImplementedError("PostgresObservationStore.fetch_by_ids is not implemented yet")
