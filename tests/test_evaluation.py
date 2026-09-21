import asyncio
from pathlib import Path

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from feesbot.chat import ChatServiceError
from feesbot.evaluation import (
    EvalCase,
    amounts_in,
    check_response,
    evaluate_retrieval,
    format_report,
    format_retrieval_report,
    load_golden,
    run_eval,
)
from feesbot.schemas import ChatResponse, SourceCitation

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "eval" / "golden.json"


def cite(bank="N26", page=5, page_end=None):
    return SourceCitation(bank=bank, document="doc.pdf", page=page, page_end=page_end)


def response(answer, grounded=True, sources=()):
    return ChatResponse(session_id="s", answer=answer, grounded=grounded, sources=list(sources))


CASE = EvalCase(
    id="c",
    question="q",
    expect_amounts=[10],
    forbid_amounts=[99],
    expect_source={"bank": "N26", "pages": [5]},
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The fee is €45.", {45.0}),
        ("45.00 €", {45.0}),
        ("45,00 EUR", {45.0}),
        ("limit of EUR 1,000 per month, 1% fee", {1000.0, 1.0}),
        ("2% of the withdrawal, minimum EUR1", {2.0, 1.0}),
        ("express delivery: approx. 1-3 business days", {1.0, 3.0}),
        ("nothing numeric here", set()),
    ],
)
def test_amounts_in(text, expected):
    assert amounts_in(text) == expected


def test_a_correct_response_passes():
    assert check_response(CASE, response("N26 charges 10.00 €.", sources=[cite(page=5, page_end=6)])) == []


def test_the_original_bug_fails_the_check():
    # 45 attributed to the standard card, citing only page 6, missing the 10 EUR on page 5
    failures = check_response(CASE, response("A standard replacement costs 45.00 €.", sources=[cite(page=6)]))
    assert any("expected amount 10" in f for f in failures)
    assert any("citation covering page(s) [5]" in f for f in failures)


def test_forbidden_amount_and_wrong_bank_fail():
    failures = check_response(CASE, response("10 or maybe 99", sources=[cite(bank="Revolut", page=5)]))
    assert any("forbidden amount 99" in f for f in failures)
    assert any("no N26 citation" in f for f in failures)


def test_unsupported_wording_is_flagged_case_insensitively():
    case = EvalCase(id="w", question="q", forbid_text=["on top of"])
    answer = "Express delivery is charged ON TOP OF the base fee."
    assert any("unsupported wording" in f for f in check_response(case, response(answer)))
    assert check_response(case, response("Express replacement costs 30.00 EUR.")) == []


def test_grounded_mismatch_is_reported():
    refusal_case = EvalCase(id="r", question="q", expect_grounded=False)
    assert check_response(refusal_case, response("no info", grounded=False)) == []
    assert "expected grounded=False, got True" in check_response(refusal_case, response("It is 5."))[0]


def test_a_refusal_case_cannot_expect_facts():
    with pytest.raises(ValidationError):
        EvalCase(id="r", question="q", expect_grounded=False, expect_amounts=[5])


def test_golden_file_is_valid_and_ids_unique():
    cases = load_golden(GOLDEN_PATH)
    assert len(cases) >= 8
    assert {"n26_standard_replacement", "n26_metal_replacement", "refuse_usd_conversion"} <= {c.id for c in cases}
    assert all(len(c.variants) >= 1 for c in cases)


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, question):
        return self.docs


def passage(text, bank="N26", page=4, page_end=None):
    meta = {"bank": bank, "source": "N26/13account-pricelist-en.pdf", "page": page}
    if page_end is not None:
        meta["page_end"] = page_end
    return Document(page_content=text, metadata=meta)


RETRIEVAL_CASE = EvalCase(
    id="std",
    question="q",
    paraphrases=["q2"],
    expect_source={"bank": "N26", "pages": [5]},
    evidence={"bank": "N26", "contains": ["Ordering a replacement card", "10.00"]},
)


def test_retrieval_passes_when_the_evidence_is_retrieved_on_the_right_page():
    good = passage("Ordering a replacement card\n10.00 EUR", page=4)  # 0-indexed page 4 = page 5
    results = evaluate_retrieval(FakeRetriever([good]), [RETRIEVAL_CASE])
    assert [r.passed for r in results] == [True, True]  # question and its paraphrase


def test_retrieval_fails_when_the_evidence_is_missing_the_wrong_bank_or_the_wrong_page():
    missing = passage("Express delivery of a replacement card\n30.00 EUR")
    wrong_bank = passage("Ordering a replacement card\n10.00 EUR", bank="Revolut")
    wrong_page = passage("Ordering a replacement card\n10.00 EUR", page=9)
    assert not evaluate_retrieval(FakeRetriever([missing]), [RETRIEVAL_CASE])[0].found
    assert not evaluate_retrieval(FakeRetriever([wrong_bank]), [RETRIEVAL_CASE])[0].found
    result = evaluate_retrieval(FakeRetriever([wrong_page]), [RETRIEVAL_CASE])[0]
    assert result.found and not result.pages_ok and not result.passed


def test_retrieval_report_names_what_was_retrieved_for_failures():
    results = evaluate_retrieval(FakeRetriever([passage("30.00 EUR")]), [RETRIEVAL_CASE])
    report = format_retrieval_report(results)
    assert "FAIL" in report and "evidence not in any retrieved passage" in report and "0/2" in report


class FakeService:
    """Answers correctly, except for one deliberately wrong paraphrase and one failing call."""

    def __init__(self):
        self.sessions = []

    async def ask(self, request):
        self.sessions.append(request.session_id)
        if request.question == "boom":
            raise ChatServiceError("llm down")
        if request.question == "wrong":
            return response("It costs 45.00 €.", sources=[cite(page=6)])
        return response("It costs 10.00 €.", sources=[cite(page=5)])


def test_run_eval_uses_fresh_sessions_and_reports_each_variant():
    good = EvalCase(id="good", question="ok", paraphrases=["ok too"], expect_amounts=[10], expect_source={"bank": "N26", "pages": [5]})
    flaky = EvalCase(id="flaky", question="ok", paraphrases=["wrong", "boom"], expect_amounts=[10])
    service = FakeService()

    results = asyncio.run(run_eval(service, [good, flaky]))

    assert len(set(service.sessions)) == len(service.sessions) == 5  # no session reuse
    assert results[0].passed and not results[1].passed
    assert [v.passed for v in results[1].variants] == [True, False, False]
    assert results[1].variants[1].failures and results[1].variants[1].error is None  # a wrong answer
    assert results[1].variants[2].error == "llm down" and not results[1].variants[2].failures  # a provider error
    report = format_report(results)
    assert "PASS  good" in report and "FAIL  flaky" in report and "1/2 cases passed" in report
    assert "provider error: llm down" in report and "1 question(s) were NOT evaluated" in report


def test_a_case_with_only_provider_errors_is_error_not_fail():
    case = EvalCase(id="e", question="boom")
    (result,) = asyncio.run(run_eval(FakeService(), [case]))
    assert result.status == "ERROR" and not result.passed
    assert "ERROR e" in format_report([result])


def test_delay_is_applied_between_questions_but_not_before_the_first(monkeypatch):
    pauses = []

    async def fake_sleep(seconds):
        pauses.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    cases = [EvalCase(id="a", question="ok", paraphrases=["ok too"]), EvalCase(id="b", question="ok")]
    asyncio.run(run_eval(FakeService(), cases, delay_s=7.5))
    assert pauses == [7.5, 7.5]  # 3 questions -> 2 pauses
