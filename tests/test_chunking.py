import re
from pathlib import Path

import pymupdf
import pytest

from feesbot.chunking import chunk_pdf, extract_pdf, pdf_files
from feesbot.evaluation import load_golden
from feesbot.hybrid import build_windows, index_corpus
from feesbot.schemas import Bank, ChunkMetadata
from feesbot.settings import Settings

ROOT = Path(__file__).resolve().parent.parent


def make_pdf(path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((50, 60), text, fontsize=9)
    doc.save(path)
    doc.close()
    return path


def test_page_offsets_cover_every_page(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["first page", "second page", "third page"])
    text, starts = extract_pdf(pdf)
    assert len(starts) == 3 and starts[0] == 0
    assert text[starts[1]:].startswith("second page")
    assert text[starts[2]:].startswith("third page")


def test_a_row_split_across_a_page_break_stays_in_one_chunk(tmp_path):
    # heading and price at the bottom of page 1, description at the top of page 2
    pdf = make_pdf(tmp_path / "a.pdf", ["Other Fees\nOrdering a widget\n10.00 EUR", "The fee is charged for lost widgets."])
    chunks = chunk_pdf(pdf, Bank.N26, chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert "10.00 EUR" in chunks[0].page_content and "lost widgets" in chunks[0].page_content
    assert (chunks[0].metadata["page"], chunks[0].metadata["page_end"]) == (0, 1)


def test_chunks_are_mapped_to_the_right_page(tmp_path):
    filler = "\n".join(f"filler line number {i:02d}" for i in range(40))
    pdf = make_pdf(tmp_path / "a.pdf", [filler, "PAGE_TWO_MARKER here\n" + filler.replace("filler", "other")])
    chunks = chunk_pdf(pdf, Bank.REVOLUT, chunk_size=150, chunk_overlap=20)
    marker = next(c for c in chunks if "PAGE_TWO_MARKER" in c.page_content)
    assert marker.metadata["page"] in (0, 1) and marker.metadata["page_end"] == 1
    first = chunks[0]
    assert (first.metadata["page"], first.metadata["page_end"]) == (0, 0)
    assert chunks[-1].metadata["page_end"] == 1


def test_every_chunk_page_range_matches_where_its_text_came_from(tmp_path):
    # each line is tagged with its page, so the true page range of any chunk can be read off its text
    pages = ["\n".join(f"p{n}line{i:02d} lorem ipsum dolor" for i in range(25)) for n in (1, 2, 3)]
    pdf = make_pdf(tmp_path / "a.pdf", pages)
    chunks = chunk_pdf(pdf, Bank.N26, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 6
    for chunk in chunks:
        true_pages = {int(n) - 1 for n in re.findall(r"p(\d)line", chunk.page_content)}
        assert (chunk.metadata["page"], chunk.metadata["page_end"]) == (min(true_pages), max(true_pages)), chunk.page_content
    assert any(c.metadata["page"] == 2 == c.metadata["page_end"] for c in chunks)  # some chunks live wholly on page 3
    assert any(c.metadata["page"] < c.metadata["page_end"] for c in chunks)  # and some span a page break


def test_every_chunk_has_valid_metadata(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["alpha beta\n" * 30, "gamma delta\n" * 30])
    for chunk in chunk_pdf(pdf, Bank.N26, 200, 30):
        meta = ChunkMetadata.model_validate(chunk.metadata)
        assert meta.bank is Bank.N26 and meta.source.endswith("a.pdf")
        assert "start_index" not in chunk.metadata


def test_pdf_files_is_sorted_and_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ("b.pdf", "sub/a.pdf"):
        make_pdf(tmp_path / name, ["x"])
    (tmp_path / "ignore.txt").write_text("x")
    files = pdf_files(tmp_path)
    assert files == sorted(files) and {p.name for p in files} == {"a.pdf", "b.pdf"}
    single = tmp_path / "b.pdf"
    assert pdf_files(single) == [single]


def test_chunk_index_counts_from_zero_in_reading_order(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["alpha beta\n" * 30, "gamma delta\n" * 30])
    chunks = chunk_pdf(pdf, Bank.N26, 200, 30)
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


# ---- the real documents: every verified fact must be retrievable as one passage, on the right page ----
# A retrieved passage is a chunk plus its neighbours (see feesbot.hybrid), so that is what must hold the fact.

SETTINGS = Settings(_env_file=None, groq_api_key="x")
GOLDEN = [c for c in load_golden(ROOT / "eval" / "golden.json") if c.evidence]
_corpus_cache: dict[Bank, dict] = {}


def bank_corpus(bank: Bank):
    if bank not in _corpus_cache:
        source = ROOT / SETTINGS.bank_sources[bank]
        _corpus_cache[bank] = index_corpus(
            [
                c
                for pdf in pdf_files(source)
                for c in chunk_pdf(pdf, bank, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
            ]
        )
    return _corpus_cache[bank]


@pytest.mark.parametrize("case", GOLDEN, ids=lambda c: c.id)
def test_verified_fact_sits_in_one_passage_on_the_right_page(case):
    source = ROOT / SETTINGS.bank_sources[case.evidence.bank]
    if not source.exists():
        pytest.skip(f"source documents not present: {source}")

    corpus = bank_corpus(case.evidence.bank)
    windows = [build_windows([key], corpus, SETTINGS.neighbor_window)[0] for key in corpus]
    matches = [w for w in windows if all(t in w.page_content for t in case.evidence.contains)]
    assert matches, f"no passage holds {case.evidence.contains} together: the fact was split apart"

    if case.expect_source:
        citations = [ChunkMetadata.model_validate(w.metadata).to_citation() for w in matches]
        assert any(
            cit.covers(p) for cit in citations for p in case.expect_source.pages
        ), f"passage pages {[c.label for c in citations]} do not cover {case.expect_source.pages}"
