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

"""Compare memory2 SQLite and PostgreSQL backend performance.

Examples:
    uv run scripts/postgres_vs_sqlite_performance_test.py --sizes 1000,10000
    uv run scripts/postgres_vs_sqlite_performance_test.py \
        --postgres-dsn postgresql://dimos:dimos@localhost:5432/dimos \
        --sizes 1000,10000,50000 --queries 200 --json-out /tmp/memory2_perf.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
from types import TracebackType
from uuid import uuid4

import numpy as np

from dimos.memory2.store.base import Store
from dimos.memory2.store.postgres import PostgresStore
from dimos.memory2.store.sqlite import SqliteStore
from dimos.models.embedding.base import Embedding

try:
    import psycopg
except ImportError:  # pragma: no cover - depends on local extras
    psycopg = None  # type: ignore[assignment]


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    size: int
    benchmark: str
    samples: int
    seconds_min: float | None
    seconds_median: float | None
    ops_per_second: float | None
    status: str
    detail: str = ""


class BackendContext(AbstractContextManager["BackendContext"]):
    name: str
    store: Store

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self.store.stop()


class SqliteBackendContext(BackendContext):
    def __init__(self, path: Path, *, keep: bool) -> None:
        self.name = "sqlite"
        self.path = path
        self.keep = keep
        self.store = SqliteStore(path=str(path))

    def close(self) -> None:
        super().close()
        if not self.keep and self.path.exists():
            self.path.unlink()


class PostgresBackendContext(BackendContext):
    def __init__(self, dsn: str, schema: str, *, keep: bool) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg is not installed; cannot benchmark PostgreSQL")
        self.name = "postgres"
        self.schema = schema
        self.keep = keep
        self.conn = psycopg.connect(dsn)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA public")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
        self.conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        self.conn.execute(f'SET search_path TO "{schema}"')
        self.conn.commit()
        self.store = PostgresStore(conn=self.conn)

    def close(self) -> None:
        super().close()
        if not self.keep:
            self.conn.execute("SET search_path TO public")
            self.conn.execute(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE')
            self.conn.commit()
        self.conn.close()


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _normalize_rows(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.maximum(norms, 1e-10)).astype(np.float32)


def _embedding(row: np.ndarray) -> Embedding:
    return Embedding(vector=row.astype(np.float32, copy=False))


def _time_samples(operation: Callable[[], int], repeats: int) -> tuple[list[float], int]:
    samples: list[float] = []
    last_ops = 0
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        last_ops = operation()
        samples.append(time.perf_counter() - start)
    return samples, last_ops


def _result(
    backend: str,
    size: int,
    benchmark: str,
    samples: list[float],
    operations: int,
    *,
    status: str = "OK",
    detail: str = "",
) -> BenchmarkResult:
    seconds_min = min(samples) if samples else None
    seconds_median = statistics.median(samples) if samples else None
    ops_per_second = None
    if seconds_min is not None and seconds_min > 0:
        ops_per_second = operations / seconds_min
    return BenchmarkResult(
        backend=backend,
        size=size,
        benchmark=benchmark,
        samples=len(samples),
        seconds_min=seconds_min,
        seconds_median=seconds_median,
        ops_per_second=ops_per_second,
        status=status,
        detail=detail,
    )


def _error_result(
    backend: str,
    size: int,
    benchmark: str,
    exc: BaseException,
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=backend,
        size=size,
        benchmark=benchmark,
        samples=0,
        seconds_min=None,
        seconds_median=None,
        ops_per_second=None,
        status="ERROR",
        detail=f"{type(exc).__name__}: {exc}",
    )


def _run_timed(
    backend: str,
    size: int,
    benchmark: str,
    operation: Callable[[], int],
    *,
    repeats: int,
) -> BenchmarkResult:
    try:
        samples, operations = _time_samples(operation, repeats)
    except BaseException as exc:
        return _error_result(backend, size, benchmark, exc)
    return _result(backend, size, benchmark, samples, operations)


def _prepare_scalar_stream(store: Store, size: int, stream_name: str) -> BenchmarkResult:
    rng = np.random.default_rng(size)
    positions = rng.uniform(-100.0, 100.0, size=(size, 3)).astype(np.float64)
    stream = store.stream(stream_name, int)

    def insert() -> int:
        for i, (x, y, z) in enumerate(positions):
            stream.append(
                i,
                ts=float(i),
                pose=(float(x), float(y), float(z), 0.0, 0.0, 0.0, 1.0),
                tags={"bucket": i % 10, "kind": "even" if i % 2 == 0 else "odd"},
            )
        return size

    return _run_timed(store_name(store), size, "insert_scalar_pose_tags", insert, repeats=1)


def _prepare_vector_stream(
    store: Store,
    size: int,
    stream_name: str,
    *,
    vector_dim: int,
) -> tuple[BenchmarkResult, list[Embedding]]:
    rng = np.random.default_rng(size + vector_dim)
    vectors = _normalize_rows(rng.normal(size=(size, vector_dim)).astype(np.float32))
    embeddings = [_embedding(row) for row in vectors]
    stream = store.stream(stream_name, int)

    def insert() -> int:
        for i, emb in enumerate(embeddings):
            stream.append(i, ts=float(i), embedding=emb)
        return size

    result = _run_timed(store_name(store), size, f"insert_vectors_dim_{vector_dim}", insert, repeats=1)
    queries = [_embedding(row) for row in vectors[: min(len(vectors), 256)]]
    return result, queries


def _basic_query_benchmarks(
    store: Store,
    size: int,
    stream_name: str,
    *,
    repeats: int,
) -> list[BenchmarkResult]:
    stream = store.stream(stream_name)
    backend = store_name(store)
    mid = size / 2.0
    return [
        _run_timed(
            backend,
            size,
            "count_all",
            lambda: stream.count(),
            repeats=repeats,
        ),
        _run_timed(
            backend,
            size,
            "scan_all",
            lambda: len(stream.to_list()),
            repeats=repeats,
        ),
        _run_timed(
            backend,
            size,
            "time_range_count",
            lambda: stream.time_range(mid - size * 0.1, mid + size * 0.1).count(),
            repeats=repeats,
        ),
        _run_timed(
            backend,
            size,
            "tag_filter_count",
            lambda: stream.tags(bucket=3).count(),
            repeats=repeats,
        ),
        _run_timed(
            backend,
            size,
            "order_by_ts_desc_limit_100",
            lambda: len(stream.order_by("ts", desc=True).limit(100).to_list()),
            repeats=repeats,
        ),
    ]


def _near_benchmarks(
    store: Store,
    size: int,
    stream_name: str,
    radii: list[float],
    *,
    queries: int,
    repeats: int,
) -> list[BenchmarkResult]:
    stream = store.stream(stream_name)
    backend = store_name(store)
    rng = np.random.default_rng(size * 11)
    centers = rng.uniform(-100.0, 100.0, size=(queries, 3)).astype(np.float64)
    results: list[BenchmarkResult] = []

    for radius in radii:
        label = str(radius).replace(".", "_")

        def count_near(radius: float = radius) -> int:
            total = 0
            for x, y, z in centers:
                total += stream.near((float(x), float(y), float(z)), radius).count()
            return queries

        def fetch_near(radius: float = radius) -> int:
            total = 0
            for x, y, z in centers:
                total += len(stream.near((float(x), float(y), float(z)), radius).to_list())
            return queries if total >= 0 else queries

        results.append(
            _run_timed(
                backend,
                size,
                f"near_count_radius_{label}",
                count_near,
                repeats=repeats,
            )
        )
        results.append(
            _run_timed(
                backend,
                size,
                f"near_fetch_radius_{label}",
                fetch_near,
                repeats=repeats,
            )
        )

    return results


def _vector_benchmarks(
    store: Store,
    size: int,
    stream_name: str,
    query_embeddings: list[Embedding],
    top_ks: list[int],
    *,
    queries: int,
    repeats: int,
) -> list[BenchmarkResult]:
    stream = store.stream(stream_name)
    backend = store_name(store)
    selected = query_embeddings[:queries]
    results: list[BenchmarkResult] = []

    for top_k in top_ks:

        def search(top_k: int = top_k) -> int:
            for query in selected:
                stream.search(query, k=top_k).to_list()
            return len(selected)

        results.append(
            _run_timed(
                backend,
                size,
                f"vector_search_top_{top_k}",
                search,
                repeats=repeats,
            )
        )

    return results


def store_name(store: Store) -> str:
    if isinstance(store, SqliteStore):
        return "sqlite"
    if isinstance(store, PostgresStore):
        return "postgres"
    return type(store).__name__


def _backend_contexts(args: argparse.Namespace) -> list[BackendContext]:
    contexts: list[BackendContext] = []
    requested = set(args.backends)

    if "sqlite" in requested:
        sqlite_path = Path(args.sqlite_path) if args.sqlite_path else _default_sqlite_path()
        if sqlite_path.exists() and not args.replace_sqlite:
            raise FileExistsError(
                f"SQLite benchmark path already exists: {sqlite_path}. "
                "Use --replace-sqlite or pass a different --sqlite-path."
            )
        if sqlite_path.exists():
            sqlite_path.unlink()
        contexts.append(SqliteBackendContext(sqlite_path, keep=args.keep))

    if "postgres" in requested:
        dsn = args.postgres_dsn or os.environ.get("POSTGRES_DSN") or os.environ.get(
            "DIMOS_POSTGRES_DSN"
        )
        if not dsn:
            raise ValueError(
                "PostgreSQL benchmark requested but no DSN was provided. "
                "Use --postgres-dsn, POSTGRES_DSN, or DIMOS_POSTGRES_DSN."
            )
        schema = args.postgres_schema or f"memory2_perf_{uuid4().hex}"
        contexts.append(PostgresBackendContext(dsn, schema, keep=args.keep))

    return contexts


def _default_sqlite_path() -> Path:
    return Path(tempfile.gettempdir()) / f"memory2_perf_{uuid4().hex}.db"


def _run_backend(
    context: BackendContext,
    sizes: list[int],
    args: argparse.Namespace,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    for size in sizes:
        prefix = f"perf_{size}_{uuid4().hex[:8]}"
        scalar_stream = f"{prefix}_scalar"
        vector_stream = f"{prefix}_vector"

        scalar_result = _prepare_scalar_stream(context.store, size, scalar_stream)
        results.append(scalar_result)
        if scalar_result.status == "OK":
            results.extend(
                _basic_query_benchmarks(
                    context.store,
                    size,
                    scalar_stream,
                    repeats=args.repeats,
                )
            )
            results.extend(
                _near_benchmarks(
                    context.store,
                    size,
                    scalar_stream,
                    args.near_radii,
                    queries=args.queries,
                    repeats=args.repeats,
                )
            )

        vector_result, query_embeddings = _prepare_vector_stream(
            context.store,
            size,
            vector_stream,
            vector_dim=args.vector_dim,
        )
        results.append(vector_result)
        if vector_result.status == "OK":
            results.extend(
                _vector_benchmarks(
                    context.store,
                    size,
                    vector_stream,
                    query_embeddings,
                    args.top_k,
                    queries=args.queries,
                    repeats=args.repeats,
                )
            )

    return results


def _print_results(results: list[BenchmarkResult]) -> None:
    headers = [
        "backend",
        "size",
        "benchmark",
        "samples",
        "min_s",
        "median_s",
        "ops_s",
        "status",
        "detail",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result.backend,
                str(result.size),
                result.benchmark,
                str(result.samples),
                _fmt_float(result.seconds_min),
                _fmt_float(result.seconds_median),
                _fmt_float(result.ops_per_second),
                result.status,
                result.detail,
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

    print("  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.3f}"
    return f"{value:.6f}"


def _write_json(path: Path, results: list[BenchmarkResult]) -> None:
    path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["sqlite", "postgres"],
        default=["sqlite"],
        help="Backends to benchmark. Pass both names to compare both.",
    )
    parser.add_argument(
        "--sqlite-path",
        help="SQLite benchmark DB path. Defaults to a unique file under the temp directory.",
    )
    parser.add_argument(
        "--replace-sqlite",
        action="store_true",
        help="Delete --sqlite-path if it already exists.",
    )
    parser.add_argument(
        "--postgres-dsn",
        help="PostgreSQL DSN. Also read from POSTGRES_DSN or DIMOS_POSTGRES_DSN.",
    )
    parser.add_argument(
        "--postgres-schema",
        help="Schema to use for PostgreSQL benchmark. Defaults to a unique temporary schema.",
    )
    parser.add_argument(
        "--sizes",
        default="1000,10000",
        help="Comma-separated row counts to benchmark.",
    )
    parser.add_argument("--vector-dim", type=int, default=128, help="Embedding dimensionality.")
    parser.add_argument(
        "--top-k",
        default="1,10,100",
        help="Comma-separated vector search k values.",
    )
    parser.add_argument(
        "--near-radii",
        default="1.0,5.0,20.0",
        help="Comma-separated radii for spatial near tests.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of vector and near queries per sample.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeated samples for read/query benchmarks. Inserts run once.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep generated SQLite DB / PostgreSQL schema after the run.",
    )
    args = parser.parse_args()
    args.sizes = _parse_ints(args.sizes)
    args.top_k = _parse_ints(args.top_k)
    args.near_radii = _parse_floats(args.near_radii)
    if not args.sizes:
        parser.error("--sizes must contain at least one integer")
    if args.vector_dim <= 0:
        parser.error("--vector-dim must be positive")
    if args.queries <= 0:
        parser.error("--queries must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    return args


def main() -> None:
    args = parse_args()
    results: list[BenchmarkResult] = []
    contexts = _backend_contexts(args)
    try:
        for context in contexts:
            print(f"Running {context.name} benchmarks...")
            results.extend(_run_backend(context, args.sizes, args))
    finally:
        for context in contexts:
            context.close()

    _print_results(results)
    if args.json_out is not None:
        _write_json(args.json_out, results)
        print(f"\nWrote JSON results to {args.json_out}")


if __name__ == "__main__":
    main()
