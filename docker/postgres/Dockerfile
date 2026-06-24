ARG FROM_IMAGE=pgvector/pgvector:pg16-bookworm
FROM ${FROM_IMAGE}

ARG POSTGIS_MAJOR=3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-${PG_MAJOR}-postgis-${POSTGIS_MAJOR} \
    postgresql-${PG_MAJOR}-postgis-${POSTGIS_MAJOR}-scripts \
    && rm -rf /var/lib/apt/lists/*

COPY docker/postgres/initdb/10-extensions.sql /docker-entrypoint-initdb.d/10-extensions.sql
