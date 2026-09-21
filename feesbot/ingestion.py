"""Build and load the persistent vector index. Run separately from serving requests."""

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from feesbot.chunking import chunk_pdf, pdf_files
from feesbot.fingerprint import index_fingerprint
from feesbot.schemas import ChunkMetadata
from feesbot.settings import Settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "bank_fees"
FINGERPRINT_FILE = "index.fingerprint"


def load_chunks(settings: Settings) -> list[Document]:
    """Read every configured PDF and return validated, page-tagged chunks.

    Raises:
        FileNotFoundError: if a configured source is missing or holds no PDFs.
        ValidationError: if a chunk ends up without valid metadata.
    """
    chunks: list[Document] = []
    for bank, path in settings.bank_sources.items():
        if not path.exists():
            raise FileNotFoundError(f"Source for {bank.value} not found: {path}")
        files = pdf_files(path)
        if not files:
            raise FileNotFoundError(f"No PDF files for {bank.value} in {path}")
        bank_chunks: list[Document] = []
        for pdf in files:
            bank_chunks.extend(chunk_pdf(pdf, bank, settings.chunk_size, settings.chunk_overlap))
        for chunk in bank_chunks:
            ChunkMetadata.model_validate(chunk.metadata)  # fail at ingestion, not at answer time
        logger.info("%s: %d PDFs -> %d chunks", bank.value, len(files), len(bank_chunks))
        chunks.extend(bank_chunks)
    return chunks


def _open_store(settings: Settings) -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model, model_kwargs={"device": "cpu"}
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def get_vectorstore(settings: Settings, rebuild: bool = False) -> Chroma:
    """Open the persisted index, rebuilding it if it is empty, stale, or `rebuild` is set."""
    fingerprint = index_fingerprint(settings)
    fingerprint_path = settings.chroma_dir / FINGERPRINT_FILE

    store = _open_store(settings)
    existing = fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
    if rebuild or existing != fingerprint:
        if store.get(limit=1)["ids"]:
            logger.info("Index is %s; rebuilding", "being rebuilt on request" if rebuild else "out of date")
        store.delete_collection()
        store = _open_store(settings)

    if not store.get(limit=1)["ids"]:
        chunks = load_chunks(settings)
        store.add_documents(chunks)
        fingerprint_path.write_text(fingerprint)
        logger.info("Indexed %d chunks into %s", len(chunks), settings.chroma_dir)
    return store
