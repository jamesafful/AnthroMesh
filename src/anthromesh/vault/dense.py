from __future__ import annotations

import hashlib
import json
import math
from typing import Callable

from anthromesh.models import DecisionLog

from .postgres import PostgresBehavioralVault
from .storage import BehavioralVault


def vector_table(dimensions: int) -> str:
    if type(dimensions) is not int or not 1 <= dimensions <= 2000:
        raise ValueError("Indexed embeddings must contain 1 to 2000 dimensions.")
    return f"dense_vectors_{dimensions}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_sqlite_vec(conn) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def delete_indexed_decisions(conn, decision_ids: list[str], *, postgres: bool) -> None:
    for decision_id in decision_ids:
        records = conn.execute(
            "SELECT id, index_name FROM dense_records WHERE decision_id = ?", (decision_id,),
        ).fetchall()
        if records and not postgres:
            _load_sqlite_vec(conn)
        for record in records:
            dimensions = int(record["index_name"].split(":", 1)[0])
            table = vector_table(dimensions)
            conn.execute(f"DELETE FROM {table} WHERE id = ?", (record["id"],))
        conn.execute("DELETE FROM dense_records WHERE decision_id = ?", (decision_id,))


class DenseDecisionIndex:
    """Cache document embeddings in the vault and query a scoped vector index."""

    def __init__(
        self, vault: BehavioralVault, index_model: str,
        embed_batch: Callable[[list[str]], list[list[float]]],
    ) -> None:
        self.vault = vault
        self.model_key = _digest(index_model)
        self.embed_batch = embed_batch
        self.postgres = isinstance(vault, PostgresBehavioralVault)

    def _create_index(self, conn, table: str, dimensions: int) -> None:
        if self.postgres:
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                id BIGINT PRIMARY KEY, scope TEXT NOT NULL, embedding public.vector({dimensions}) NOT NULL
            )""")
            conn.execute(f"""CREATE INDEX IF NOT EXISTS {table}_cosine
                ON {table} USING hnsw (embedding public.vector_cosine_ops)""")
        else:
            _load_sqlite_vec(conn)
            conn.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
                id INTEGER PRIMARY KEY, scope TEXT PARTITION KEY,
                embedding FLOAT[{dimensions}] distance_metric=cosine
            )""")

    @staticmethod
    def _validate(vector: list[float], dimensions: int) -> None:
        if (len(vector) != dimensions or any(not math.isfinite(item) for item in vector)
                or not any(vector)):
            raise ValueError("Indexed embeddings must have matching dimensions and finite, nonzero values.")

    def _encoded(self, vector: list[float]):
        if self.postgres:
            return json.dumps(vector, separators=(",", ":"))
        from sqlite_vec import serialize_float32

        return serialize_float32(vector)

    def rank(
        self, query_text: str, documents: list[tuple[DecisionLog, str]], *, limit: int,
    ) -> dict[str, float]:
        if not documents or limit <= 0:
            return {}
        vectors = self.embed_batch([query_text])
        if len(vectors) != 1:
            raise ValueError("Embedding backend returned an invalid query batch.")
        query = vectors[0]
        dimensions = len(query)
        table = vector_table(dimensions)
        self._validate(query, dimensions)
        index_name = f"{dimensions}:{self.model_key}"
        first = documents[0][0]
        scope = _digest(json.dumps([index_name, first.subject_id, first.domain]))
        if any((item.subject_id, item.domain) != (first.subject_id, first.domain)
               for item, _ in documents):
            raise ValueError("Indexed retrieval requires a single subject and domain.")
        fingerprint = {item.decision_id: _digest(text) for item, text in documents}
        slots = ", ".join("?" for _ in fingerprint)
        read_records = f"""SELECT id, decision_id, content_digest FROM dense_records
                           WHERE index_name = ? AND decision_id IN ({slots})"""
        record_args = (index_name, *fingerprint)
        with self.vault._connect() as conn:
            records = conn.execute(read_records, record_args).fetchall()
        cached = {row["decision_id"]: row for row in records if row["decision_id"] in fingerprint}
        pending = [(item, text) for item, text in documents
                   if item.decision_id not in cached
                   or cached[item.decision_id]["content_digest"] != fingerprint[item.decision_id]]
        prepared = []
        for offset in range(0, len(pending), 64):
            batch = pending[offset:offset + 64]
            values = self.embed_batch([text for _, text in batch])
            if len(values) != len(batch):
                raise ValueError("Embedding backend returned an invalid document batch.")
            for (item, _), vector in zip(batch, values):
                self._validate(vector, dimensions)
                prepared.append((item, self._encoded(vector)))

        with self.vault._connect() as conn:
            if prepared:
                conn.execute("BEGIN IMMEDIATE")
                self._create_index(conn, table, dimensions)
                for item, vector in prepared:
                    live = conn.execute(
                        "SELECT payload_json FROM decisions WHERE decision_id = ?", (item.decision_id,),
                    ).fetchone()
                    if live is None or DecisionLog.model_validate_json(live["payload_json"]) != item:
                        continue
                    row = conn.execute(
                        """INSERT INTO dense_records(index_name, decision_id, content_digest)
                           VALUES (?, ?, ?) ON CONFLICT(index_name, decision_id) DO UPDATE SET
                           content_digest = excluded.content_digest RETURNING id""",
                        (index_name, item.decision_id, fingerprint[item.decision_id]),
                    ).fetchone()
                    conn.execute(f"DELETE FROM {table} WHERE id = ?", (row["id"],))
                    value = "?::public.vector" if self.postgres else "?"
                    conn.execute(
                        f"INSERT INTO {table}(id, scope, embedding) VALUES (?, ?, {value})",
                        (row["id"], scope, vector),
                    )
            elif not self.postgres:
                _load_sqlite_vec(conn)
            exact = conn.execute(read_records, record_args).fetchall()
            ids = {row["id"]: row["decision_id"] for row in exact
                   if fingerprint.get(row["decision_id"]) == row["content_digest"]}
            if not ids:
                return {}
            slots = ", ".join("?" for _ in ids)
            query_vector = self._encoded(query)
            count = min(limit, len(ids))
            if self.postgres:
                sql = f"""SELECT id, embedding OPERATOR(public.<=>) ?::public.vector AS distance
                          FROM {table} WHERE scope = ? AND id IN ({slots})
                          ORDER BY distance LIMIT ?"""
            else:
                sql = f"""SELECT id, distance FROM {table}
                          WHERE embedding MATCH ? AND scope = ? AND id IN ({slots}) AND k = ?
                          ORDER BY distance"""
            args = (query_vector, scope, *ids, count)
            rows = conn.execute(sql, args).fetchall()
            if self.postgres and len(rows) < count:
                conn.raw.execute("SET LOCAL enable_indexscan = off")
                rows = conn.execute(sql, args).fetchall()
        return {ids[row["id"]]: max(0.0, min(1.0, 1.0 - row["distance"])) for row in rows}
