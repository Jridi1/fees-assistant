from pathlib import Path

from feesbot.fingerprint import index_fingerprint
from feesbot.schemas import Bank
from feesbot.settings import Settings


def settings_for(tmp_path: Path, **overrides) -> Settings:
    folder = tmp_path / "docs"
    folder.mkdir(exist_ok=True)
    (folder / "a.pdf").write_bytes(b"%PDF-1.4 aaa")
    return Settings(
        _env_file=None,
        groq_api_key="x",
        bank_sources={Bank.N26: folder, Bank.REVOLUT: tmp_path / "missing.pdf"},
        **overrides,
    )


def test_fingerprint_is_stable(tmp_path):
    assert index_fingerprint(settings_for(tmp_path)) == index_fingerprint(settings_for(tmp_path))


def test_fingerprint_changes_when_chunking_changes(tmp_path):
    base = index_fingerprint(settings_for(tmp_path))
    assert index_fingerprint(settings_for(tmp_path, chunk_size=800, chunk_overlap=100)) != base
    assert index_fingerprint(settings_for(tmp_path, embedding_model="other/model")) != base


def test_fingerprint_changes_when_documents_change(tmp_path):
    settings = settings_for(tmp_path)
    before = index_fingerprint(settings)
    (tmp_path / "docs" / "b.pdf").write_bytes(b"%PDF-1.4 new document")
    assert index_fingerprint(settings) != before

    after_add = index_fingerprint(settings)
    (tmp_path / "docs" / "a.pdf").write_bytes(b"%PDF-1.4 aaa but longer")
    assert index_fingerprint(settings) != after_add
