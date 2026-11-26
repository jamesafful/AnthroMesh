from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any, Iterator

from .storage import VAULT_SCHEMA, BehavioralVault


def _postgres_sql(statement: str) -> str:
    # These expressions occur in shared internal queries; values remain bound parameters.
    statement = re.sub(
        r"json_extract\(payload_json, '\$\.([a-z_.]+)'\)",
        lambda match: f"(payload_json::jsonb #>> '{{{match[1].replace('.', ',')}}}')",
        statement,
    )
    statement = statement.replace("julianday(created_at)", "CAST(created_at AS timestamptz)")
    statement = statement.replace("julianday(?)", "CAST(? AS timestamptz)")
    return statement.replace("?", "%s")


class _Connection:
    def __init__(self, connection: Any, lock_name: str) -> None:
        self.raw = connection
        self.lock_name = lock_name
        self.writer_locked = False

    def lock_writer(self) -> None:
        if not self.writer_locked:
            self.raw.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (self.lock_name,))
            self.writer_locked = True

    def execute(self, statement: str, parameters: Any = None) -> Any:
        command = statement.lstrip().split(None, 1)[0].upper()
        if statement == "BEGIN IMMEDIATE":
            self.lock_writer()
            return None
        if statement == "BEGIN":
            return self.raw.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        if command in {"INSERT", "UPDATE", "DELETE", "CREATE", "ALTER"}:
            self.lock_writer()
        return self.raw.execute(_postgres_sql(statement), parameters)


class PostgresBehavioralVault(BehavioralVault):
    """PostgreSQL implementation of the vault API, isolated in a configured schema."""

    def __init__(self, dsn: str, *, schema: str = "anthromesh") -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,62}", schema):
            raise ValueError("PostgreSQL schema must be a lowercase SQL identifier.")
        if schema in {"public", "information_schema"} or schema.startswith("pg_"):
            raise ValueError("AnthroMesh requires its own PostgreSQL schema.")
        import psycopg

        options = psycopg.conninfo.conninfo_to_dict(dsn)
        if "sslmode" not in options:
            options["sslmode"] = "verify-full"
        options.setdefault("connect_timeout", "5")
        self._dsn = psycopg.conninfo.make_conninfo(**options)
        self._schema = schema
        self.db_path = f"PostgreSQL schema {schema}"
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[_Connection]:
        import psycopg
        from psycopg import sql
        from psycopg.rows import dict_row

        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            conn.execute(sql.SQL("SET LOCAL search_path TO {}, pg_catalog").format(
                sql.Identifier(self._schema)
            ))
            yield _Connection(conn, f"anthromesh:{self._schema}:writer")

    def _init_schema(self) -> None:
        from psycopg import sql

        ddl = VAULT_SCHEMA.replace(
            "INTEGER PRIMARY KEY AUTOINCREMENT", "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY"
        ).replace("confidence REAL", "confidence DOUBLE PRECISION")
        with self._connect() as conn:
            conn.lock_writer()
            schema_exists = conn.execute(
                "SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = ?", (self._schema,),
            ).fetchone()
            if not schema_exists:
                conn.raw.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self._schema)))
            for statement in ddl.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.execute("ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS entry_hash TEXT")
