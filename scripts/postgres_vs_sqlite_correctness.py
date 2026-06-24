#!/usr/bin/env python3
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

"""Run identical memory2 operations against SQLite and PostgreSQL and compare results.

Examples:
    uv run scripts/postgres_vs_sqlite_correctness.py \
        --postgres-dsn postgresql://dimos:dimos@localhost:5432/dimos

    POSTGRES_DSN=postgresql://dimos:dimos@localhost:5432/dimos \
        uv run scripts/postgres_vs_sqlite_correctness.py --json-out /tmp/memory2_correctness.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from types import TracebackType
from typing import Any
from uuid import uuid4

import numpy as np

from dimos.memory2.store.base import Store
from dimos.memory2.store.postgres import PostgresStore
from dimos.memory2.store.sqlite import SqliteStore
from dimos.memory2.type.observation import Observation
from dimos.models.embedding.base import Embedding

try:
    import psycopg
except ImportError:  # pragma: no cover - depends on installed extras
    psycopg = None  # type: ignore[assignment]


JsonValue = dict[str, Any] | list[Any] | str | int | float | bool | None


@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    sqlite: JsonValue
    postgres: JsonValue
    detail: str = ""


class StoreHandle(AbstractContextManager["StoreHandle"]):
    def __init__(self, store: Store, close_extra: Callable[[], None] | None = None) -> None:
        self.store = store
        self._close_extra = close_extra

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.store.stop()
        if self._close_extra is not None:
            self._close_extra()


class TestDatabases:
    def __init__(
        self,
        *,
        sqlite_path: Path,
        postgres_dsn: str,
        postgres_schema: str,
        keep: bool,
    ) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg is not installed; cannot run PostgreSQL correctness tests")
        self.sqlite_path = sqlite_path
        self.postgres_dsn = postgres_dsn
        self.postgres_schema = postgres_schema
        self.keep = keep
        self._setup_postgres()

    def _setup_postgres(self) -> None:
        conn = psycopg.connect(self.postgres_dsn)
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public")
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            conn.execute(f'DROP SCHEMA IF EXISTS "{self.postgres_schema}" CASCADE')
            conn.execute(f'CREATE SCHEMA "{self.postgres_schema}"')
            conn.commit()
        finally:
            conn.close()

    def open_sqlite(self) -> StoreHandle:
        return StoreHandle(SqliteStore(path=str(self.sqlite_path)))

    def open_postgres(self) -> StoreHandle:
        conn = psycopg.connect(self.postgres_dsn)
        conn.execute(f'SET search_path TO "{self.postgres_schema}"')
        conn.commit()
        return StoreHandle(PostgresStore(conn=conn), close_extra=conn.close)

    def cleanup(self) -> None:
        if not self.keep and self.sqlite_path.exists():
            self.sqlite_path.unlink()
        if self.keep:
            return
        conn = psycopg.connect(self.postgres_dsn)
        try:
            conn.execute(f'DROP SCHEMA IF EXISTS "{self.postgres_schema}" CASCADE')
            conn.commit()
        finally:
            conn.close()


def _embedding(values: list[float]) -> Embedding:
    arr = np.array(values, dtype=np.float32)
    arr = arr / (np.linalg.norm(arr) + 1e-10)
    return Embedding(vector=arr)


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _normalize_value(value: Any) -> JsonValue:
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return repr(value)


def _normalize_obs(obs: Observation[Any]) -> dict[str, JsonValue]:
    normalized: dict[str, JsonValue] = {
        "data": _normalize_value(obs.data),
        "data_type": type(obs.data).__name__,
        "ts": _round_float(obs.ts),
        "pose": _normalize_value(obs.pose_tuple),
        "tags": _normalize_value(obs.tags),
    }
    similarity = getattr(obs, "similarity", None)
    if similarity is not None:
        normalized["similarity"] = _round_float(float(similarity))
    return normalized


def _normalize_observations(observations: list[Observation[Any]]) -> list[dict[str, JsonValue]]:
    return [_normalize_obs(obs) for obs in observations]


def _capture(operation: Callable[[Store], JsonValue], store: Store) -> JsonValue:
    try:
        return {"ok": operation(store)}
    except BaseException as exc:
        return {"error": type(exc).__name__}


def _equivalent(left: JsonValue, right: JsonValue, *, tolerance: float) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_equivalent(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_equivalent(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True))
    return left == right


def _populate(store: Store) -> JsonValue:
    numbers = store.stream("numbers", int)
    numbers.append(1, ts=10.0, pose=(0, 0, 0), tags={"kind": "odd", "bucket": 1})
    numbers.append(2, ts=20.0, pose=(1, 0, 0), tags={"kind": "even", "bucket": 2})
    numbers.append(3, ts=30.0, pose=(5, 0, 0), tags={"kind": "odd", "bucket": 1})
    numbers.append(4, ts=40.0, pose=(0, 5, 0), tags={"kind": "even", "bucket": 3})
    numbers.append(5, ts=50.0, tags={"kind": "missing_pose", "bucket": 3})

    floats = store.stream("floats", float)
    floats.append(1.25, ts=1.0)
    floats.append(2.5, ts=2.0)
    floats.append(3.75, ts=3.0)

    logs = store.stream("logs", str)
    logs.append("alpha", ts=1.0, tags={"level": "info"})
    logs.append("beta", ts=2.0, tags={"level": "warn"})
    logs.append("gamma", ts=3.0, tags={"level": "info"})

    vectors = store.stream("vectors", str)
    vectors.append("north", ts=1.0, tags={"axis": "y"}, embedding=_embedding([0.0, 1.0, 0.0]))
    vectors.append("east", ts=2.0, tags={"axis": "x"}, embedding=_embedding([1.0, 0.0, 0.0]))
    vectors.append("south", ts=3.0, tags={"axis": "y"}, embedding=_embedding([0.0, -1.0, 0.0]))
    vectors.append("northeast", ts=4.0, tags={"axis": "xy"}, embedding=_embedding([1.0, 2.0, 0.0]))

    bulk = store.stream("bulk_numbers", int)
    bulk.append_many(
        [10, 20, 30],
        ts=[100.0, 200.0, 300.0],
        pose=[(10, 0, 0), (20, 0, 0), None],
        tags=[{"batch": 1}, {"batch": 1}, {"batch": 2}],
    )

    return sorted(store.list_streams())


def _cases() -> list[tuple[str, Callable[[Store], JsonValue]]]:
    return [
        ("list_streams", lambda store: sorted(store.list_streams())),
        ("numbers_all", lambda store: _normalize_observations(store.stream("numbers").to_list())),
        (
            "numbers_after",
            lambda store: _normalize_observations(store.stream("numbers").after(15.0).to_list()),
        ),
        (
            "numbers_before",
            lambda store: _normalize_observations(store.stream("numbers").before(35.0).to_list()),
        ),
        (
            "numbers_time_range",
            lambda store: _normalize_observations(
                store.stream("numbers").time_range(15.0, 40.0).to_list()
            ),
        ),
        (
            "numbers_at",
            lambda store: _normalize_observations(store.stream("numbers").at(30.25, 0.5).to_list()),
        ),
        (
            "numbers_tags_single",
            lambda store: _normalize_observations(store.stream("numbers").tags(kind="odd").to_list()),
        ),
        (
            "numbers_tags_multi",
            lambda store: _normalize_observations(
                store.stream("numbers").tags(kind="odd", bucket=1).to_list()
            ),
        ),
        (
            "numbers_near_small",
            lambda store: _normalize_observations(
                store.stream("numbers").near((0, 0, 0), 1.5).to_list()
            ),
        ),
        (
            "numbers_near_large",
            lambda store: _normalize_observations(
                store.stream("numbers").near((0, 0, 0), 6.0).to_list()
            ),
        ),
        (
            "numbers_order_offset_limit",
            lambda store: _normalize_observations(
                store.stream("numbers").order_by("ts", desc=True).offset(1).limit(2).to_list()
            ),
        ),
        ("numbers_count_all", lambda store: store.stream("numbers").count()),
        ("numbers_count_near", lambda store: store.stream("numbers").near((0, 0, 0), 6.0).count()),
        (
            "bulk_numbers_all",
            lambda store: _normalize_observations(store.stream("bulk_numbers").to_list()),
        ),
        (
            "bulk_numbers_near",
            lambda store: _normalize_observations(
                store.stream("bulk_numbers").near((15, 0, 0), 6.0).to_list()
            ),
        ),
        ("floats_all", lambda store: _normalize_observations(store.stream("floats").to_list())),
        ("logs_all", lambda store: _normalize_observations(store.stream("logs").to_list())),
        (
            "logs_tag_filter",
            lambda store: _normalize_observations(store.stream("logs").tags(level="info").to_list()),
        ),
        (
            "vector_search_north_top_3",
            lambda store: _normalize_observations(
                store.stream("vectors").search(_embedding([0.0, 1.0, 0.0]), k=3).to_list()
            ),
        ),
        (
            "vector_search_east_top_2",
            lambda store: _normalize_observations(
                store.stream("vectors").search(_embedding([1.0, 0.0, 0.0]), k=2).to_list()
            ),
        ),
        (
            "vector_search_with_limit",
            lambda store: _normalize_observations(
                store.stream("vectors")
                .search(_embedding([0.0, 1.0, 0.0]), k=4)
                .limit(2)
                .to_list()
            ),
        ),
    ]


def _compare(
    name: str,
    sqlite_value: JsonValue,
    postgres_value: JsonValue,
    *,
    tolerance: float,
) -> CaseResult:
    passed = _equivalent(sqlite_value, postgres_value, tolerance=tolerance)
    return CaseResult(name=name, passed=passed, sqlite=sqlite_value, postgres=postgres_value)


def _run_case_set(
    dbs: TestDatabases,
    *,
    prefix: str,
    tolerance: float,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    with dbs.open_sqlite() as sqlite_handle, dbs.open_postgres() as postgres_handle:
        for name, operation in _cases():
            sqlite_value = _capture(operation, sqlite_handle.store)
            postgres_value = _capture(operation, postgres_handle.store)
            results.append(
                _compare(
                    f"{prefix}.{name}",
                    sqlite_value,
                    postgres_value,
                    tolerance=tolerance,
                )
            )
    return results


def _run_delete_case(dbs: TestDatabases, *, tolerance: float) -> list[CaseResult]:
    results: list[CaseResult] = []
    with dbs.open_sqlite() as sqlite_handle, dbs.open_postgres() as postgres_handle:

        def delete_logs(store: Store) -> JsonValue:
            store.delete_stream("logs")
            return sorted(store.list_streams())

        sqlite_value = _capture(delete_logs, sqlite_handle.store)
        postgres_value = _capture(delete_logs, postgres_handle.store)
        results.append(_compare("delete.logs", sqlite_value, postgres_value, tolerance=tolerance))

        def open_deleted_without_type(store: Store) -> JsonValue:
            store.stream("logs")
            return "unexpected-success"

        sqlite_value = _capture(open_deleted_without_type, sqlite_handle.store)
        postgres_value = _capture(open_deleted_without_type, postgres_handle.store)
        results.append(
            _compare(
                "delete.open_deleted_without_type",
                sqlite_value,
                postgres_value,
                tolerance=tolerance,
            )
        )

    return results


def run_correctness(args: argparse.Namespace) -> list[CaseResult]:
    sqlite_path = Path(args.sqlite_path) if args.sqlite_path else _default_sqlite_path()
    if sqlite_path.exists() and not args.replace_sqlite:
        raise FileExistsError(
            f"SQLite correctness path already exists: {sqlite_path}. "
            "Use --replace-sqlite or pass a different --sqlite-path."
        )
    if sqlite_path.exists():
        sqlite_path.unlink()

    postgres_dsn = args.postgres_dsn or os.environ.get("POSTGRES_DSN") or os.environ.get(
        "DIMOS_POSTGRES_DSN"
    )
    if not postgres_dsn:
        raise ValueError("Provide --postgres-dsn, POSTGRES_DSN, or DIMOS_POSTGRES_DSN")

    postgres_schema = args.postgres_schema or f"memory2_correctness_{uuid4().hex}"
    dbs = TestDatabases(
        sqlite_path=sqlite_path,
        postgres_dsn=postgres_dsn,
        postgres_schema=postgres_schema,
        keep=args.keep,
    )

    try:
        results: list[CaseResult] = []
        with dbs.open_sqlite() as sqlite_handle, dbs.open_postgres() as postgres_handle:
            sqlite_value = _capture(_populate, sqlite_handle.store)
            postgres_value = _capture(_populate, postgres_handle.store)
            results.append(
                _compare("populate", sqlite_value, postgres_value, tolerance=args.tolerance)
            )

        results.extend(_run_case_set(dbs, prefix="reopen", tolerance=args.tolerance))
        results.extend(_run_delete_case(dbs, tolerance=args.tolerance))
        return results
    finally:
        dbs.cleanup()


def _default_sqlite_path() -> Path:
    return Path(tempfile.gettempdir()) / f"memory2_correctness_{uuid4().hex}.db"


def _print_results(results: list[CaseResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.name}")
        if not result.passed:
            print(f"  sqlite:   {json.dumps(result.sqlite, sort_keys=True)}")
            print(f"  postgres: {json.dumps(result.postgres, sort_keys=True)}")


def _write_json(path: Path, results: list[CaseResult]) -> None:
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postgres-dsn",
        help="PostgreSQL DSN. Also read from POSTGRES_DSN or DIMOS_POSTGRES_DSN.",
    )
    parser.add_argument(
        "--postgres-schema",
        help="Schema to use for PostgreSQL correctness data. Defaults to a unique temporary schema.",
    )
    parser.add_argument(
        "--sqlite-path",
        help="SQLite correctness DB path. Defaults to a unique file under the temp directory.",
    )
    parser.add_argument(
        "--replace-sqlite",
        action="store_true",
        help="Delete --sqlite-path first if it already exists.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Float tolerance for normalized similarity and pose comparisons.",
    )
    parser.add_argument("--json-out", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep generated SQLite DB / PostgreSQL schema after the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_correctness(args)
    _print_results(results)
    if args.json_out is not None:
        _write_json(args.json_out, results)
        print(f"\nWrote JSON results to {args.json_out}")
    failed = [result for result in results if not result.passed]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
