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

from decimal import Decimal
import json
import re
import threading
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import Field

from dimos.memory2.observationstore.base import ObservationStore, ObservationStoreConfig
from dimos.memory2.type.filter import (
    AfterFilter,
    AtFilter,
    BeforeFilter,
    NearFilter,
    TagsFilter,
    TimeRangeFilter,
)
from dimos.memory2.type.observation import _UNLOADED, Observation, PoseTuple
from dimos.memory2.utils.validation import validate_identifier

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dimos.memory2.type.filter import Filter, StreamQuery

T = TypeVar("T")

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _reconstruct_pose(
    x: float | None,
    y: float | None,
    z: float | None,
    qx: float | None,
    qy: float | None,
    qz: float | None,
    qw: float | None,
) -> PoseTuple | None:
    if x is None:
        return None
    assert y is not None and z is not None
    assert qx is not None and qy is not None and qz is not None and qw is not None
    return (x, y, z, qx, qy, qz, qw)


def _json_obj(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)  # type: ignore[no-any-return]
    return dict(raw)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, Decimal):
        integral = value.to_integral_value()
        return int(integral) if value == integral else float(value)
    return value


def _compile_filter(f: Filter, prefix: str = "") -> tuple[str, list[Any]] | None:
    if isinstance(f, AfterFilter):
        return (f"{prefix}ts > %s", [f.t])
    if isinstance(f, BeforeFilter):
        return (f"{prefix}ts < %s", [f.t])
    if isinstance(f, TimeRangeFilter):
        return (f"{prefix}ts >= %s AND {prefix}ts <= %s", [f.t1, f.t2])
    if isinstance(f, AtFilter):
        return (f"ABS({prefix}ts - %s) <= %s", [f.t, f.tolerance])
    if isinstance(f, TagsFilter):
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in f.tags.items():
            if not _IDENT_RE.match(key):
                raise ValueError(f"Invalid tag key: {key!r}")
            clauses.append(f"{prefix}tags -> '{key}' = %s::jsonb")
            params.append(json.dumps(value))
        return (" AND ".join(clauses), params)
    if isinstance(f, NearFilter):
        cx, cy, cz = f.position.x, f.position.y, f.position.z
        point = "public.ST_SetSRID(public.ST_MakePoint(%s, %s, %s), 0)::public.geometry"
        return (
            f"{prefix}pose_point OPERATOR(public.&&&) public.ST_Expand({point}, %s) "
            f"AND public.ST_3DDWithin({prefix}pose_point, {point}, %s)",
            [cx, cy, cz, f.radius, cx, cy, cz, f.radius],
        )
    return None


def _compile_query(query: StreamQuery, table: str) -> tuple[str, list[Any], list[Filter]]:
    select = (
        "SELECT id, ts, value, pose_x, pose_y, pose_z, "
        f'pose_qx, pose_qy, pose_qz, pose_qw, tags FROM "{table}"'
    )
    where_parts: list[str] = []
    params: list[Any] = []
    python_filters: list[Filter] = []

    for f in query.filters:
        compiled = _compile_filter(f)
        if compiled is None:
            python_filters.append(f)
        else:
            sql_part, sql_params = compiled
            where_parts.append(sql_part)
            params.extend(sql_params)

    sql = select
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)

    if query.order_field:
        if not _IDENT_RE.match(query.order_field):
            raise ValueError(f"Invalid order_field: {query.order_field!r}")
        direction = "DESC" if query.order_desc else "ASC"
        sql += f" ORDER BY {query.order_field} {direction}"
    else:
        sql += " ORDER BY id ASC"

    if not python_filters:
        if query.limit_val is not None:
            sql += " LIMIT %s"
            params.append(query.limit_val)
            if query.offset_val:
                sql += " OFFSET %s"
                params.append(query.offset_val)
        elif query.offset_val:
            sql += " OFFSET %s"
            params.append(query.offset_val)

    return (sql, params, python_filters)


def _compile_count(query: StreamQuery, table: str) -> tuple[str, list[Any], list[Filter]]:
    where_parts: list[str] = []
    params: list[Any] = []
    python_filters: list[Filter] = []

    for f in query.filters:
        compiled = _compile_filter(f)
        if compiled is None:
            python_filters.append(f)
        else:
            sql_part, sql_params = compiled
            where_parts.append(sql_part)
            params.extend(sql_params)

    sql = f'SELECT COUNT(*) FROM "{table}"'
    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    return (sql, params, python_filters)


class PostgresObservationStoreConfig(ObservationStoreConfig):
    conn: Any = Field(exclude=True)
    name: str
    page_size: int = Field(default=256, exclude=True)


class PostgresObservationStore(ObservationStore[T]):
    """Postgres-backed metadata store for a single stream."""

    config: PostgresObservationStoreConfig

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        validate_identifier(self.config.name)
        self._conn = self.config.conn
        self._name = self.config.name
        self._page_size = self.config.page_size
        self._lock = threading.Lock()
        self._tag_indexes: set[str] = set()
        self._pending_python_filters: list[Any] = []
        self._pending_query: StreamQuery | None = None

    def start(self) -> None:
        self._ensure_table()

    def _ensure_table(self) -> None:
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public")
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{self._name}" (
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
        self._conn.execute(
            f'ALTER TABLE "{self._name}" '
            "ADD COLUMN IF NOT EXISTS pose_point public.geometry(PointZ, 0)"
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{self._name}_ts_idx" ON "{self._name}" (ts)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{self._name}_tags_idx" '
            f'ON "{self._name}" USING gin (tags)'
        )
        self._conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{self._name}_pose_point_idx" '
            f'ON "{self._name}" USING gist (pose_point public.gist_geometry_ops_nd) '
            "WHERE pose_point IS NOT NULL"
        )
        self.commit()

    @property
    def name(self) -> str:
        return self._name

    def insert(self, obs: Observation[T]) -> int:
        with self._lock:
            if obs.tags:
                self._ensure_tag_indexes(obs.tags)
            row = self._conn.execute(
                f"""
                INSERT INTO "{self._name}"
                    (
                        ts, value,
                        pose_x, pose_y, pose_z,
                        pose_qx, pose_qy, pose_qz, pose_qw,
                        pose_point,
                        tags
                    )
                VALUES (
                    %s, %s,
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
                RETURNING id
                """,
                self._insert_params(obs),
            ).fetchone()
        assert row is not None
        return int(row["id"] if isinstance(row, dict) else row[0])

    def insert_many(self, observations: list[Observation[T]]) -> list[int]:
        if not observations:
            return []
        with self._lock:
            for obs in observations:
                if obs.tags:
                    self._ensure_tag_indexes(obs.tags)
            row = self._conn.execute(
                f"""
                WITH rows AS (
                    SELECT *
                    FROM unnest(
                        %s::bigint[],
                        %s::double precision[],
                        %s::numeric[],
                        %s::double precision[],
                        %s::double precision[],
                        %s::double precision[],
                        %s::double precision[],
                        %s::double precision[],
                        %s::double precision[],
                        %s::double precision[],
                        %s::boolean[],
                        %s::jsonb[]
                    ) AS t(
                        ord, ts, value,
                        pose_x, pose_y, pose_z,
                        pose_qx, pose_qy, pose_qz, pose_qw,
                        has_pose, tags
                    )
                )
                INSERT INTO "{self._name}"
                    (
                        ts, value,
                        pose_x, pose_y, pose_z,
                        pose_qx, pose_qy, pose_qz, pose_qw,
                        pose_point,
                        tags
                    )
                SELECT
                    ts, value,
                    pose_x, pose_y, pose_z,
                    pose_qx, pose_qy, pose_qz, pose_qw,
                    CASE
                        WHEN has_pose THEN public.ST_SetSRID(
                            public.ST_MakePoint(pose_x, pose_y, pose_z),
                            0
                        )::public.geometry(PointZ, 0)
                        ELSE NULL
                    END,
                    tags
                FROM rows
                ORDER BY ord
                RETURNING id
                """,
                self._insert_arrays(observations),
            ).fetchall()
            return [int(item["id"] if isinstance(item, dict) else item[0]) for item in row]

    @staticmethod
    def _insert_params(obs: Observation[T]) -> tuple[Any, ...]:
        pose = obs.pose_tuple
        tags_json = json.dumps(obs.tags) if obs.tags else "{}"
        if isinstance(obs._data, bool):
            value = int(obs._data)
        else:
            value = obs._data if isinstance(obs._data, (int, float)) else None
        if pose:
            px, py, pz, qx, qy, qz, qw = pose
        else:
            px = py = pz = qx = qy = qz = qw = None
        return (
            obs.ts,
            value,
            px,
            py,
            pz,
            qx,
            qy,
            qz,
            qw,
            pose is not None,
            px,
            py,
            pz,
            tags_json,
        )

    @staticmethod
    def _insert_arrays(observations: list[Observation[T]]) -> tuple[list[Any], ...]:
        ords: list[int] = []
        timestamps: list[float] = []
        values: list[Decimal | None] = []
        pose_x: list[float | None] = []
        pose_y: list[float | None] = []
        pose_z: list[float | None] = []
        pose_qx: list[float | None] = []
        pose_qy: list[float | None] = []
        pose_qz: list[float | None] = []
        pose_qw: list[float | None] = []
        has_pose: list[bool] = []
        tags: list[str] = []

        for index, obs in enumerate(observations):
            ords.append(index)
            timestamps.append(float(obs.ts))
            if isinstance(obs._data, bool):
                values.append(Decimal(int(obs._data)))
            elif isinstance(obs._data, int):
                values.append(Decimal(obs._data))
            elif isinstance(obs._data, float):
                values.append(Decimal(str(obs._data)))
            else:
                values.append(None)

            pose = obs.pose_tuple
            has_pose.append(pose is not None)
            if pose is None:
                pose_x.append(None)
                pose_y.append(None)
                pose_z.append(None)
                pose_qx.append(None)
                pose_qy.append(None)
                pose_qz.append(None)
                pose_qw.append(None)
            else:
                px, py, pz, qx, qy, qz, qw = pose
                pose_x.append(px)
                pose_y.append(py)
                pose_z.append(pz)
                pose_qx.append(qx)
                pose_qy.append(qy)
                pose_qz.append(qz)
                pose_qw.append(qw)
            tags.append(json.dumps(obs.tags) if obs.tags else "{}")

        return (
            ords,
            timestamps,
            values,
            pose_x,
            pose_y,
            pose_z,
            pose_qx,
            pose_qy,
            pose_qz,
            pose_qw,
            has_pose,
            tags,
        )

    def _ensure_tag_indexes(self, tags: dict[str, Any]) -> None:
        for key in tags:
            if key not in self._tag_indexes and _IDENT_RE.match(key):
                self._conn.execute(
                    f'CREATE INDEX IF NOT EXISTS "{self._name}_tag_{key}" '
                    f'ON "{self._name}" ((tags -> \'{key}\'))'
                )
                self._tag_indexes.add(key)

    @staticmethod
    def _row_get(row: Any, key: str, idx: int) -> Any:
        if isinstance(row, dict):
            return row[key]
        return row[idx]

    def _row_to_obs(self, row: Any) -> Observation[T]:
        row_id = self._row_get(row, "id", 0)
        ts = self._row_get(row, "ts", 1)
        value = self._row_get(row, "value", 2)
        px = self._row_get(row, "pose_x", 3)
        py = self._row_get(row, "pose_y", 4)
        pz = self._row_get(row, "pose_z", 5)
        qx = self._row_get(row, "pose_qx", 6)
        qy = self._row_get(row, "pose_qy", 7)
        qz = self._row_get(row, "pose_qz", 8)
        qw = self._row_get(row, "pose_qw", 9)
        tags = _json_obj(self._row_get(row, "tags", 10))
        pose = _reconstruct_pose(px, py, pz, qx, qy, qz, qw)
        data: Any = _decode_scalar(value) if value is not None else _UNLOADED
        return Observation(id=int(row_id), ts=float(ts), pose_tuple=pose, tags=tags, _data=data)

    def query(self, q: StreamQuery) -> Iterator[Observation[T]]:
        if q.search_text is not None:
            raise NotImplementedError("search_text is not supported by PostgresObservationStore")

        sql, params, python_filters = _compile_query(q, self._name)
        cur = self._conn.execute(sql, params)
        cur.arraysize = self._page_size
        self._pending_python_filters = python_filters
        self._pending_query = q
        return (self._row_to_obs(row) for row in cur)

    def count(self, q: StreamQuery) -> int:
        if q.search_vec:
            raise NotImplementedError("count with search_vec must go through Backend")

        sql, params, python_filters = _compile_count(q, self._name)
        if python_filters:
            return sum(1 for _ in self.query(q))
        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return 0
        return int(row["count"] if isinstance(row, dict) else row[0])

    def fetch_by_ids(self, ids: list[int]) -> list[Observation[T]]:
        if not ids:
            return []
        rows = self._conn.execute(
            f"""
            SELECT id, ts, value, pose_x, pose_y, pose_z, pose_qx, pose_qy, pose_qz, pose_qw, tags
            FROM "{self._name}"
            WHERE id = ANY(%s)
            """,
            (ids,),
        ).fetchall()
        return [self._row_to_obs(row) for row in rows]

    def optimize(self) -> None:
        self._conn.execute(f'ANALYZE "{self._name}"')

    def commit(self) -> None:
        commit = getattr(self._conn, "commit", None)
        if commit is not None:
            commit()

    def rollback(self) -> None:
        rollback = getattr(self._conn, "rollback", None)
        if rollback is not None:
            rollback()
