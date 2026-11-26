from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from anthromesh.models import DecisionLog, QueryContext

if TYPE_CHECKING:
    from .dense import DenseDecisionIndex


TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "was", "were", "be", "this", "that", "it", "as", "at", "by", "from", "we", "i", "should",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class HybridDecisionRetriever:
    """Ranks decisions with BM25, optional dense similarity, and recency-weighted RRF."""

    def __init__(
        self,
        embedder: Callable[[str], list[float]] | None = None,
        *,
        embed_batch: Callable[[list[str]], list[list[float]]] | None = None,
        dense_index: DenseDecisionIndex | None = None,
    ) -> None:
        self.dense_index = dense_index
        self.embed_batch = embed_batch
        if embedder is not None and embed_batch is None:
            self.embed_batch = lambda texts: [embedder(text) for text in texts]

    @staticmethod
    def _doc_text(d: DecisionLog) -> str:
        options_text = " ".join(f"{o.option_id} {o.label} {o.description}" for o in d.options_considered)
        tags_text = " ".join(d.tags)
        return f"{d.domain} {d.context} {options_text} {d.chosen_action} {d.rationale} {tags_text}"

    @staticmethod
    def _query_text(q: QueryContext) -> str:
        options_text = " ".join(f"{o.option_id} {o.label} {o.description}" for o in q.candidate_options)
        return f"{q.domain} {q.context} {options_text}"

    @staticmethod
    def _cosine_dense(v1: list[float], v2: list[float]) -> float:
        if len(v1) != len(v2) or not all(math.isfinite(x) for x in (*v1, *v2)):
            raise ValueError("Embedding vectors must be finite and have equal dimensions.")
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (n1 * n2)))

    def retrieve(
        self,
        query: QueryContext,
        candidates: list[DecisionLog],
        top_k: int = 5,
    ) -> list[tuple[DecisionLog, float]]:
        if not candidates or top_k <= 0:
            return []

        documents = [self._doc_text(item) for item in candidates]
        query_text = self._query_text(query)
        term_counts = [Counter(tokenize(document)) for document in documents]
        lengths = [sum(counts.values()) for counts in term_counts]
        avg_length = sum(lengths) / len(lengths) or 1.0
        frequencies = Counter(term for counts in term_counts for term in counts)
        idf = {
            term: math.log1p((len(candidates) - frequencies[term] + 0.5)
                             / (frequencies[term] + 0.5))
            for term in set(tokenize(query_text))
        }
        query_weight = sum(idf.values()) or 1.0
        lexical = []
        query_options = {option.option_id.lower() for option in query.candidate_options}
        for item, counts, length in zip(candidates, term_counts, lengths):
            length_norm = 1.2 * (0.25 + 0.75 * length / avg_length)
            score = sum(weight * 2.2 * counts[term] / (counts[term] + length_norm)
                        for term, weight in idf.items())
            options = {option.option_id.lower() for option in item.options_considered}
            overlap = len(options & query_options) / max(len(options | query_options), 1)
            domain = item.domain == query.domain
            lexical.append(0.62 * min(1.0, score / query_weight) + 0.23 * overlap + 0.15 * domain)

        channels = [lexical]
        if self.dense_index is not None:
            matches = self.dense_index.rank(
                query_text, list(zip(candidates, documents)), limit=max(20, 4 * top_k),
            )
            channels.append([matches.get(item.decision_id, 0.0) for item in candidates])
        elif self.embed_batch is not None:
            vectors = self.embed_batch([query_text, *documents])
            if len(vectors) != len(documents) + 1 or not vectors[0]:
                raise ValueError("Embedding backend returned an invalid batch.")
            channels.append([self._cosine_dense(vectors[0], vector) for vector in vectors[1:]])

        fused = [0.0] * len(candidates)
        for scores in channels:
            ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], candidates[i].decision_id))
            rank = 0
            previous_score = None
            for position, i in enumerate(ranked, start=1):
                if scores[i] <= 0:
                    break
                if scores[i] != previous_score:
                    rank = position
                    previous_score = scores[i]
                fused[i] += 1.0 / (60 + rank)

        now = datetime.now(timezone.utc)
        for i, item in enumerate(candidates):
            try:
                timestamp = datetime.fromisoformat(item.created_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            days = max(0.0, (now - timestamp).total_seconds() / 86400)
            fused[i] *= 0.75 + 0.25 * 2 ** (-days / 180)

        relevance = [max(scores) for scores in zip(*channels)]
        ranked = sorted(range(len(candidates)), key=lambda i: (
            -fused[i], -relevance[i], candidates[i].decision_id
        ))
        return [(candidates[i], round(relevance[i], 4)) for i in ranked[:top_k]]
