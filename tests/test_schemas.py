import pytest
from pydantic import ValidationError

from feesbot.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    ChunkMetadata,
    SourceCitation,
    unique_citations,
)
from feesbot.settings import MissingSecretError, Settings


def test_chunk_metadata_becomes_a_one_indexed_citation():
    meta = ChunkMetadata(bank="N26", source="N26" + chr(92) + "13account-pricelist-en.pdf", page=5, total_pages=20)
    citation = meta.to_citation()
    assert citation.document == "13account-pricelist-en.pdf"
    assert citation.label == "N26 — 13account-pricelist-en.pdf, page 6"


def test_page_range_is_shown_and_covered():
    citation = ChunkMetadata(bank="N26", source="a.pdf", page=4, page_end=5).to_citation()
    assert citation.label == "N26 — a.pdf, pages 5–6"
    assert citation.covers(5) and citation.covers(6) and not citation.covers(7)


def test_single_page_range_collapses():
    citation = ChunkMetadata(bank="N26", source="a.pdf", page=4, page_end=4).to_citation()
    assert citation.page_end is None and citation.label.endswith("page 5")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bank="Monzo", source="a.pdf"),
        dict(source="a.pdf"),
        dict(bank="N26", source="a.pdf", page=5, page_end=4),
        dict(bank="N26", source="a.pdf", page_end=2),
    ],
)
def test_invalid_chunk_metadata_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        ChunkMetadata(**kwargs)


def test_citations_are_deduplicated_in_order():
    a = SourceCitation(bank="N26", document="a.pdf", page=1)
    b = SourceCitation(bank="Revolut", document="b.pdf", page=2)
    assert unique_citations([a, b, a]) == [a, b]


def test_request_is_trimmed_and_validated():
    assert ChatRequest(session_id="abc-123", question="  How much? ").question == "How much?"
    for bad in [
        dict(session_id="abc", question="   "),
        dict(session_id="abc", question="x" * 501),
        dict(session_id="../etc/passwd", question="hi"),
    ]:
        with pytest.raises(ValidationError):
            ChatRequest(**bad)


def test_a_refusal_must_not_cite_sources():
    cite = SourceCitation(bank="N26", document="a.pdf", page=1)
    ChatResponse(session_id="s", answer="EUR 10", grounded=True, sources=[cite])
    ChatResponse(session_id="s", answer="Not in the documents.", grounded=False)
    with pytest.raises(ValidationError):
        ChatResponse(session_id="s", answer="No.", grounded=False, sources=[cite])


def test_chat_events_must_match_their_type():
    response = ChatResponse(session_id="s", answer="ok", grounded=True)
    assert ChatEvent.answer(response).response is response
    assert ChatEvent.status("working").message == "working"
    assert ChatEvent.error("timeout", "too slow").code == "timeout"
    for bad in [
        dict(type="answer"),  # answer without a response
        dict(type="status"),  # status without a message
        dict(type="status", message="x", response=response),  # status carrying a response
        dict(type="error"),
        dict(type="nonsense", message="x"),
    ]:
        with pytest.raises(ValidationError):
            ChatEvent(**bad)


def test_settings_validate_and_hide_secrets(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    settings = Settings(_env_file=None, groq_api_key="super-secret")
    assert "super-secret" not in repr(settings)
    assert (settings.chunk_size, settings.chunk_overlap) == (600, 90)
    assert (settings.retrieval_k, settings.neighbor_window) == (4, 1)

    # the key is optional to build the index, but required to serve
    assert Settings(_env_file=None).groq_api_key is None
    assert settings.require_groq_key().get_secret_value() == "super-secret"
    for missing in (Settings(_env_file=None), Settings(_env_file=None, groq_api_key="   ")):
        with pytest.raises(MissingSecretError, match="GROQ_API_KEY"):
            missing.require_groq_key()
    for bad in [dict(chunk_size=200, chunk_overlap=200), dict(retrieval_k=0)]:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, groq_api_key="x", **bad)
