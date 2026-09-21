"""Retriever construction."""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever

from feesbot.hybrid import HybridRetriever
from feesbot.prompts import QUERY_PROMPT
from feesbot.settings import Settings


def load_corpus(store: Chroma) -> list[Document]:
    """Every indexed chunk with its metadata, in reading order."""
    data = store.get(include=["documents", "metadatas"])
    docs = [Document(page_content=text, metadata=meta) for text, meta in zip(data["documents"], data["metadatas"])]
    return sorted(docs, key=lambda d: (d.metadata["source"], d.metadata["chunk_index"]))


def build_retriever(store: Chroma, llm: BaseChatModel | None, settings: Settings) -> BaseRetriever:
    """Hybrid retriever, optionally wrapped in multi-query expansion.

    Multi-query (off by default) asks the LLM for paraphrases of the question and merges the
    results, at the cost of one extra LLM call per question. Pass `llm=None` for retrieval
    without any LLM, e.g. to measure recall.
    """
    hybrid = HybridRetriever(
        dense_search=lambda query, n: store.similarity_search(query, k=n),
        corpus=load_corpus(store),
        k=settings.retrieval_k,
        candidates=settings.retrieval_candidates,
        window=settings.neighbor_window,
    )
    if llm is None or not settings.use_multi_query:
        return hybrid
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever

    return MultiQueryRetriever.from_llm(retriever=hybrid, llm=llm, prompt=QUERY_PROMPT)
