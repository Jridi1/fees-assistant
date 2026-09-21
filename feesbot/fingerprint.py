"""Detect when the persisted index no longer matches the documents and settings."""

import hashlib
import json

from feesbot.chunking import pdf_files
from feesbot.settings import Settings


def index_fingerprint(settings: Settings) -> str:
    """Hash of everything that determines the index contents.

    Changes when the PDFs (names or sizes), chunking or embedding model change, so a
    stale index is detected and rebuilt instead of silently serving outdated chunks.
    """
    sources = {
        bank.value: [(str(f), f.stat().st_size) for f in pdf_files(path)]
        for bank, path in settings.bank_sources.items()
        if path.exists()
    }
    payload = {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embedding_model": settings.embedding_model,
        "sources": sources,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
