"""Turn PDFs into overlapping chunks that keep rows and paragraphs together across page breaks.

Splitting page by page cuts price-list rows in half when they cross a page break
(the heading and price on one page, the description on the next), which made the
model attach a price to the wrong item. Here each PDF is joined into one text
first, split once, and every chunk is mapped back to the page(s) it came from.
"""

from bisect import bisect_right
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from feesbot.schemas import Bank


def extract_pdf(path: Path) -> tuple[str, list[int]]:
    """Return the PDF's full text and the character offset where each page starts."""
    parts: list[str] = []
    page_starts: list[int] = []
    offset = 0
    with pymupdf.open(path) as pdf:
        for page in pdf:
            text = page.get_text().rstrip("\n") + "\n"
            page_starts.append(offset)
            parts.append(text)
            offset += len(text)
    return "".join(parts), page_starts


def chunk_pdf(
    path: Path, bank: Bank, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split one PDF into chunks tagged with bank, source file and 0-indexed page range."""
    text, page_starts = extract_pdf(path)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.create_documents([text], metadatas=[{"bank": bank.value, "source": str(path)}])
    for position, chunk in enumerate(chunks):
        start = chunk.metadata.pop("start_index")
        end = start + len(chunk.page_content) - 1
        chunk.metadata["page"] = bisect_right(page_starts, start) - 1
        chunk.metadata["page_end"] = bisect_right(page_starts, end) - 1
        chunk.metadata["chunk_index"] = position  # order within the file, used to find neighbours
    return chunks


def pdf_files(path: Path) -> list[Path]:
    """A single PDF, or every PDF under a folder (sorted, so indexing is deterministic)."""
    if path.is_dir():
        return sorted(path.rglob("*.pdf"))
    return [path]
