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
    expect_sources=[{"bank": "N26", "pages": [5]}],
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
    assert len(cases) >= 20
    assert {"n26_standard_replacement", "refuse_usd_conversion", "compare_replacement_fee"} <= {c.id for c in cases}
    assert all(len(c.variants) >= 1 for c in cases)


def test_the_golden_set_covers_every_kind_of_question_we_claim_to_handle():
    cases = {c.id: c for c in load_golden(GOLDEN_PATH)}
    assert any(c.context for c in cases.values()), "no follow-up (multi-turn) case"
    assert any(len(c.expect_sources) > 1 for c in cases.values()), "no cross-bank comparison"
    assert any(c.id.startswith("injection_") for c in cases.values()), "no prompt-injection case"
    assert any(not c.expect_grounded for c in cases.values()), "no refusal case"
    assert any(c.expect_text for c in cases.values()), "no wording (non-numeric) case"
    banks = {s.bank.value for c in cases.values() for s in c.expect_sources}
    assert banks == {"N26", "Revolut"}


def test_expected_wording_is_checked_ignoring_case_and_line_wraps():
    case = EvalCase(id="t", question="q", expect_text=["two months"])
    assert check_response(case, response("At least TWO\n  months of notice.")) == []
    assert any("expected wording" in f for f in check_response(case, response("A month.")))


def test_every_expected_source_must_be_cited():
    case = EvalCase(id="c", question="q", expect_sources=[{"bank": "N26", "pages": [5]}, {"bank": "Revolut", "pages": [2]}])
    only_n26 = response("both", sources=[cite("N26", page=5)])
    both = response("both", sources=[cite("N26", page=5), cite("Revolut", page=2)])
    assert any("Revolut citation" in f for f in check_response(case, only_n26))
    assert check_response(case, both) == []


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
    expect_sources=[{"bank": "N26", "pages": [5]}],
    evidence=[{"bank": "N26", "contains": ["Ordering a replacement card", "10.00"]}],
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


def test_retrieval_matches_evidence_split_by_a_line_wrap():
    wrapped = passage("Ordering a replacement\ncard\n10.00 EUR", page=4)
    assert evaluate_retrieval(FakeRetriever([wrapped]), [RETRIEVAL_CASE])[0].passed


COMPARISON = EvalCase(
    id="cmp",
    question="Which is cheaper?",
    expect_sources=[{"bank": "N26", "pages": [5]}, {"bank": "Revolut", "pages": [2]}],
    evidence=[
        {"bank": "N26", "contains": ["Ordering a replacement card", "10.00"]},
        {"bank": "Revolut", "contains": ["Replacement Revolut Cards", "per replacement"]},
    ],
)


def test_a_comparison_needs_the_evidence_of_every_bank_to_be_retrieved():
    n26 = passage("Ordering a replacement card\n10.00 EUR", page=4)
    revolut = passage("Replacement Revolut Cards\nEUR6 per replacement", bank="Revolut", page=1)
    assert evaluate_retrieval(FakeRetriever([n26, revolut]), [COMPARISON])[0].passed
    only_n26 = evaluate_retrieval(FakeRetriever([n26]), [COMPARISON])[0]
    assert not only_n26.found and not only_n26.passed  # a comparison that retrieves one bank cannot be answered


def test_follow_up_cases_are_skipped_by_the_retrieval_only_check():
    follow_up = EvalCase(id="f", question="and at Revolut?", context=["first"], evidence=[{"bank": "Revolut", "contains": ["x"]}])
    assert evaluate_retrieval(FakeRetriever([]), [follow_up]) == []


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
    good = EvalCase(id="good", question="ok", paraphrases=["ok too"], expect_amounts=[10], expect_sources=[{"bank": "N26", "pages": [5]}])
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


class RecordingService:
    """Records (session, question) in order; fails on the question 'boom'."""

    def __init__(self):
        self.calls = []

    async def ask(self, request):
        self.calls.append((request.session_id, request.question))
        if request.question == "boom":
            raise ChatServiceError("llm down")
        return response("It costs 6.00 €.", sources=[cite("Revolut", page=2)])


def test_context_turns_are_asked_first_in_the_same_session_and_only_the_last_is_judged():
    case = EvalCase(id="fu", context=["first", "second"], question="and Revolut?", paraphrases=["what about Revolut?"], expect_amounts=[6])
    service = RecordingService()
    (result,) = asyncio.run(run_eval(service, [case]))

    sessions = [s for s, _ in service.calls]
    assert [q for _, q in service.calls] == ["first", "second", "and Revolut?", "first", "second", "what about Revolut?"]
    assert sessions[0] == sessions[1] == sessions[2] != sessions[3] == sessions[4] == sessions[5]  # one session per variant
    assert result.passed and [v.question for v in result.variants] == ["and Revolut?", "what about Revolut?"]


def test_a_failed_context_turn_makes_the_variant_an_error_not_a_failure():
    case = EvalCase(id="fu", context=["boom"], question="and Revolut?", expect_amounts=[6])
    service = RecordingService()
    (result,) = asyncio.run(run_eval(service, [case]))
    assert result.status == "ERROR" and result.variants[0].error == "llm down"
    assert [q for _, q in service.calls] == ["boom"]  # the judged question was never asked


def test_a_known_issue_is_reported_but_does_not_fail_the_run():
    wrong = EvalCase(id="w", question="wrong", expect_amounts=[10], known_issue="documented limitation")
    right = EvalCase(id="r", question="ok", expect_amounts=[10], known_issue="documented limitation")
    plain = EvalCase(id="p", question="wrong", expect_amounts=[10])
    broken = EvalCase(id="b", question="boom", known_issue="documented limitation")
    results = {r.case_id: r for r in asyncio.run(run_eval(FakeService(), [wrong, right, plain, broken]))}

    assert (results["w"].status, results["w"].ok) == ("KNOWN", True)  # still wrong, but documented
    assert (results["r"].status, results["r"].ok) == ("FIXED", True)  # passes now: time to remove the marker
    assert (results["p"].status, results["p"].ok) == ("FAIL", False)  # an undocumented failure fails the run
    assert (results["b"].status, results["b"].ok) == ("ERROR", False)  # a provider error is never excused

    report = format_report(list(results.values()))
    assert "known issue: documented limitation" in report and "remove known_issue" in report
    assert "1 known issue (documented, not counted as failures)" in report


def test_retrieval_known_issues_are_reported_like_the_live_ones():
    known = EvalCase(id="k", question="q", evidence=[{"bank": "N26", "contains": ["never retrieved"]}], known_issue="why")
    fixed = EvalCase(id="f", question="q", evidence=[{"bank": "N26", "contains": ["holds"]}], known_issue="why")
    plain = EvalCase(id="p", question="q", evidence=[{"bank": "N26", "contains": ["never retrieved"]}])
    results = {r.case_id: r for r in evaluate_retrieval(FakeRetriever([passage("this holds")]), [known, fixed, plain])}

    assert [(results[i].status, results[i].ok) for i in "kfp"] == [("KNOWN", True), ("FIXED", True), ("FAIL", False)]
    report = format_retrieval_report(list(results.values()))
    assert "known issue: why" in report and "remove known_issue" in report and "(1 known issue, see above)" in report


def test_every_known_issue_in_the_golden_set_explains_itself():
    documented = [c for c in load_golden(GOLDEN_PATH) if c.known_issue]
    assert documented, "the set is expected to carry its honest limitations"
    for case in documented:
        assert len(case.known_issue) > 60, f"{case.id}: a known issue needs a real explanation, not a label"


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

    pauses.clear()
    asyncio.run(run_eval(FakeService(), [EvalCase(id="c", context=["a", "b"], question="q")], delay_s=1))
    assert pauses == [1, 1]  # context turns are questions too, and cost rate limit
