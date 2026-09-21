"""Hybrid retrieval: embeddings + BM25 keyword search, fused, then widened with neighbouring chunks.

Why: on fee tables, embedding search alone never surfaced the row holding the answer (a short
heading plus price buried in a longer chunk), and a bigger `k` did not help. Keyword search finds
the heading by its exact words; fusing both rankings ranks it high. Small chunks retrieve
precisely, so each hit is then expanded with its neighbours to give the LLM the whole row.
"""

import heapq
import re
from collections.abc import Callable, Hashable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr
from rank_bm25 import BM25Okapi

RRF_K = 60  # standard reciprocal-rank-fusion constant
MIN_OVERLAP = 15  # shorter shared text is treated as coincidence, not chunk overlap

ChunkKey = tuple[str, int]  # (source file, chunk_index)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens, so '10.00' matches as '10' and '00'."""
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_key(doc: Document) -> ChunkKey:
    """The (source file, chunk_index) pair that identifies a chunk."""
    return doc.metadata["source"], doc.metadata["chunk_index"]


def index_corpus(docs: Sequence[Document]) -> dict[ChunkKey, Document]:
    """Look-up table from (source, chunk_index) to chunk."""
    return {chunk_key(d): d for d in docs}


def reciprocal_rank_fusion(rankings: Sequence[Sequence[Hashable]], k: int = RRF_K) -> list[Hashable]:
    """Merge several best-first rankings into one; items ranked well by several lists win.

    Each item scores `1 / (k + rank)` in every list that contains it; scores are summed.

    Args:
        rankings: Best-first lists of items, e.g. the embedding results and the keyword results.
        k: Damping constant. Larger values flatten the difference between ranks.

    Returns:
        Every item that appeared in any list, best first.
    """
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def join_without_overlap(previous: str, following: str) -> str:
    """Concatenate two consecutive chunks, dropping the text they share from the overlap."""
    for size in range(min(len(previous), len(following)), MIN_OVERLAP - 1, -1):
        if previous.endswith(following[:size]):
            return previous + following[size:]
    return previous + "\n" + following


def build_windows(hits: Sequence[ChunkKey], corpus: dict[ChunkKey, Document], window: int) -> list[Document]:
    """Turn ranked hits into context windows.

    Each hit is widened by `window` chunks either side (within its own file). Windows that touch
    or overlap are merged, so the same text is never sent twice. Windows come back ordered by the
    best rank of the hits inside them, and carry the bank and the full page range they cover.

    Args:
        hits: Chunk keys, best first. Keys missing from `corpus` are ignored.
        corpus: Every chunk, looked up by key (see `index_corpus`).
        window: How many neighbouring chunks to add on each side of a hit.

    Returns:
        One document per merged window, with `bank`, `source`, `page`, `page_end` and `chunk_index`.
    """
    ranges: dict[str, list[list[int]]] = {}  # source -> [[lo, hi, best_rank], ...]
    for rank, (source, index) in enumerate(hits):
        if (source, index) not in corpus:
            continue
        lo, hi = max(0, index - window), index + window
        ranges.setdefault(source, []).append([lo, hi, rank])

    merged: list[tuple[str, int, int, int]] = []  # (source, lo, hi, best_rank)
    for source, spans in ranges.items():
        spans.sort()
        current = spans[0]
        for lo, hi, rank in spans[1:]:
            if lo <= current[1] + 1:
                current[1] = max(current[1], hi)
                current[2] = min(current[2], rank)
            else:
                merged.append((source, *current))
                current = [lo, hi, rank]
        merged.append((source, *current))

    documents = []
    for source, lo, hi, _ in sorted(merged, key=lambda m: m[3]):
        chunks = [corpus[(source, i)] for i in range(lo, hi + 1) if (source, i) in corpus]
        text = chunks[0].page_content
        for chunk in chunks[1:]:
            text = join_without_overlap(text, chunk.page_content)
        first = chunks[0].metadata
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "bank": first["bank"],
                    "source": source,
                    "page": min(c.metadata["page"] for c in chunks),
                    "page_end": max(c.metadata.get("page_end", c.metadata["page"]) for c in chunks),
                    "chunk_index": chunks[0].metadata["chunk_index"],
                },
            )
        )
    return documents


class HybridRetriever(BaseRetriever):
    """Embedding search + BM25, fused with reciprocal rank fusion, widened with neighbours.

    `dense_search(query, n)` must return the n nearest chunks by embedding; `corpus` is every
    indexed chunk (each with `source` and `chunk_index` metadata).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dense_search: Callable[[str, int], list[Document]]
    corpus: list[Document]
    k: int = 4  # number of top hits to expand
    candidates: int = 30  # how many results each method contributes before fusion
    window: int = 1  # neighbouring chunks added on each side

    _lookup: dict[ChunkKey, Document] = PrivateAttr(default_factory=dict)
    _keys: list[ChunkKey] = PrivateAttr(default_factory=list)
    _bm25: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Build the keyword index once, when the retriever is created."""
        self._lookup = index_corpus(self.corpus)
        self._keys = [chunk_key(d) for d in self.corpus]
        self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.corpus])

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        dense = [chunk_key(d) for d in self.dense_search(query, self.candidates)]
        scores = self._bm25.get_scores(tokenize(query))
        best = heapq.nlargest(self.candidates, range(len(scores)), key=scores.__getitem__)
        lexical = [self._keys[i] for i in best if scores[i] > 0]
        hits = reciprocal_rank_fusion([dense, lexical])[: self.k]
        return build_windows(hits, self._lookup, self.window)
