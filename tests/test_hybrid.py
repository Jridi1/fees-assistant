from langchain_core.documents import Document

from feesbot.hybrid import (
    HybridRetriever,
    build_windows,
    index_corpus,
    join_without_overlap,
    reciprocal_rank_fusion,
    tokenize,
)


def mk(index, text, source="a.pdf", page=0, page_end=None, bank="N26"):
    meta = {"bank": bank, "source": source, "page": page, "chunk_index": index}
    if page_end is not None:
        meta["page_end"] = page_end
    return Document(page_content=text, metadata=meta)


def corpus_of(*texts, source="a.pdf"):
    return [mk(i, t, source=source, page=i // 2) for i, t in enumerate(texts)]


def test_tokenize_keeps_numbers_and_ignores_case():
    assert tokenize("Ordering a Replacement card: 10.00 EUR") == ["ordering", "a", "replacement", "card", "10", "00", "eur"]


def test_fusion_favours_items_ranked_by_several_lists():
    assert reciprocal_rank_fusion([["a", "b", "c"], ["b", "d"]]) == ["b", "a", "d", "c"]


def test_overlap_between_chunks_is_not_repeated():
    shared = "the quick brown fox jumps"
    joined = join_without_overlap("start of text. " + shared, shared + " over the lazy dog")
    assert joined == "start of text. " + shared + " over the lazy dog"


def test_chunks_without_overlap_are_joined_with_a_newline():
    assert join_without_overlap("first chunk", "second chunk") == "first chunk\nsecond chunk"


def test_a_hit_is_widened_with_its_neighbours_and_page_range():
    docs = corpus_of("zero", "one", "two", "three", "four")
    (window,) = build_windows([("a.pdf", 2)], index_corpus(docs), window=1)
    assert window.page_content == "one\ntwo\nthree"
    assert window.metadata["page"] == 0 and window.metadata["page_end"] == 1  # chunks 1..3 live on pages 0..1
    assert window.metadata["bank"] == "N26" and window.metadata["chunk_index"] == 1


def test_touching_windows_are_merged_so_text_is_not_sent_twice():
    docs = corpus_of("zero", "one", "two", "three", "four")
    windows = build_windows([("a.pdf", 1), ("a.pdf", 3)], index_corpus(docs), window=1)
    assert len(windows) == 1 and windows[0].page_content == "zero\none\ntwo\nthree\nfour"


def test_distant_windows_stay_separate_and_follow_the_hit_ranking():
    docs = corpus_of("zero", "one", "two", "three", "four", "five", "six")
    windows = build_windows([("a.pdf", 6), ("a.pdf", 0)], index_corpus(docs), window=0)
    assert [w.page_content for w in windows] == ["six", "zero"]


def test_windows_are_clipped_at_the_edges_of_a_file():
    docs = corpus_of("zero", "one", "two")
    (window,) = build_windows([("a.pdf", 0)], index_corpus(docs), window=2)
    assert window.page_content == "zero\none\ntwo"


def test_neighbours_never_come_from_another_file():
    docs = corpus_of("a0", "a1", "a2", source="a.pdf") + corpus_of("b0", "b1", "b2", source="b.pdf")
    (window,) = build_windows([("a.pdf", 2)], index_corpus(docs), window=1)
    assert window.page_content == "a1\na2"


def test_unknown_hits_are_ignored():
    assert build_windows([("missing.pdf", 3)], index_corpus(corpus_of("x")), window=1) == []


def make_retriever(docs, dense_results, **kwargs):
    return HybridRetriever(dense_search=lambda query, n: dense_results[:n], corpus=docs, **kwargs)


FILLER = ["general terms apply", "account overview text", "card usage notes", "interest rate table", "closing remarks"]


def test_keyword_match_rescues_what_embedding_search_misses():
    docs = corpus_of(*FILLER, "the zebra surcharge is 10.00 EUR", "more filler words")
    retriever = make_retriever(docs, dense_results=[docs[0]], k=3, window=0)  # embeddings find nothing relevant
    contents = [d.page_content for d in retriever.invoke("what is the zebra surcharge")]
    assert "the zebra surcharge is 10.00 EUR" in contents


def test_an_embedding_only_hit_is_still_returned():
    docs = corpus_of(*FILLER)
    retriever = make_retriever(docs, dense_results=[docs[3]], k=2, window=0)
    assert docs[3].page_content in [d.page_content for d in retriever.invoke("qqqq")]


def test_k_limits_the_number_of_hits():
    docs = corpus_of(*FILLER, "zebra one", "zebra two", "zebra three")
    retriever = make_retriever(docs, dense_results=[], k=1, window=0)
    assert len(retriever.invoke("zebra")) == 1


def test_no_match_anywhere_returns_nothing():
    retriever = make_retriever(corpus_of(*FILLER), dense_results=[], k=3, window=1)
    assert retriever.invoke("qqqq") == []
