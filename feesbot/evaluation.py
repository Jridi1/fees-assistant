"""Golden-set evaluation: does the assistant give the *correct* answer, from the right page?

`grounded` only says the model found something in the documents, not that it read the
right row. Each case therefore pins down a fact verified against the source PDF: the
amounts an answer must contain, the ones it must not, and a page it must cite. Every
paraphrase of a question is asked in a fresh session and must pass the same checks, which
is what tests retrieval consistency.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from langchain_core.retrievers import BaseRetriever

from feesbot.chat import ChatService, ChatServiceError
from feesbot.schemas import Bank, ChatRequest, ChatResponse, ChunkMetadata, Question


class ExpectedSource(BaseModel):
    """A citation the answer must include: this bank, covering any of these 1-indexed pages."""

    bank: Bank
    pages: list[int] = Field(min_length=1)


class Evidence(BaseModel):
    """Strings that must all appear together in one indexed chunk of `bank` (offline check)."""

    bank: Bank
    contains: list[str] = Field(min_length=1)


class EvalCase(BaseModel):
    """One golden question and everything needed to judge an answer to it.

    A case may carry paraphrases (asked in fresh sessions, all judged the same way), earlier turns
    of the conversation (`context`, asked first in the same session and not judged), the amounts
    and wording an answer must or must not contain, the banks and pages it must cite, and
    `evidence` for the offline retrieval check. A case that expects a refusal
    (`expect_grounded=False`) cannot expect facts.
    """

    id: str = Field(min_length=1)
    question: Question
    context: list[Question] = Field(default_factory=list)  # asked first, in the same session
    paraphrases: list[Question] = Field(default_factory=list)
    expect_grounded: bool = True
    expect_amounts: list[float] = Field(default_factory=list)
    forbid_amounts: list[float] = Field(default_factory=list)
    expect_text: list[str] = Field(default_factory=list)  # phrases that must appear (any case, any spacing)
    forbid_text: list[str] = Field(default_factory=list)  # phrases that signal a known invented claim
    expect_sources: list[ExpectedSource] = Field(default_factory=list)  # every one must be cited
    evidence: list[Evidence] = Field(default_factory=list)  # every one must be retrievable
    # A documented, unresolved failure. It is reported as KNOWN and does not fail a run; once the case
    # passes it is reported as FIXED, as a reminder to delete this line.
    known_issue: str | None = None

    @model_validator(mode="after")
    def _refusal_cases_expect_nothing_else(self) -> "EvalCase":
        if not self.expect_grounded and (self.expect_amounts or self.expect_sources or self.expect_text):
            raise ValueError("a case expecting a refusal cannot expect amounts, wording or sources")
        return self

    @property
    def variants(self) -> list[str]:
        """The question and all its paraphrases."""
        return [self.question, *self.paraphrases]


def load_golden(path: Path) -> list[EvalCase]:
    """Load and validate the golden set. Raises on duplicate ids or malformed cases."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases = [EvalCase.model_validate(item) for item in raw["cases"]]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in golden set")
    return cases


_NUMBER = re.compile(r"\d[\d.,]*\d|\d")


def _to_float(token: str) -> float:
    if "," in token and "." in token:
        decimal = "," if token.rfind(",") > token.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        token = token.replace(thousands, "").replace(decimal, ".")
    elif "," in token or "." in token:
        sep = "," if "," in token else "."
        parts = token.split(sep)
        if len(parts) > 2 or len(parts[1]) == 3:  # 1,000 / 1.000 / 1,000,000 are thousands
            token = "".join(parts)
        else:
            token = token.replace(",", ".")
    return float(token)


def amounts_in(text: str) -> set[float]:
    """Every number in the text as a float, treating 45,00 / 45.00 / 1,000 sensibly."""
    return {round(_to_float(t.rstrip(".,")), 2) for t in _NUMBER.findall(text)}


def check_response(case: EvalCase, response: ChatResponse) -> list[str]:
    """Return the reasons this response fails the case (empty list means it passes)."""
    failures: list[str] = []
    if response.grounded != case.expect_grounded:
        failures.append(f"expected grounded={case.expect_grounded}, got {response.grounded}")

    found = amounts_in(response.answer)
    for amount in case.expect_amounts:
        if round(amount, 2) not in found:
            failures.append(f"answer does not contain expected amount {amount:g}")
    for amount in case.forbid_amounts:
        if round(amount, 2) in found:
            failures.append(f"answer contains forbidden amount {amount:g}")

    lowered = normalize_ws(response.answer).lower()
    for phrase in case.expect_text:
        if normalize_ws(phrase).lower() not in lowered:
            failures.append(f"answer does not contain expected wording: '{phrase}'")
    for phrase in case.forbid_text:
        if normalize_ws(phrase).lower() in lowered:
            failures.append(f"answer contains unsupported wording: '{phrase}'")

    for expected in case.expect_sources:
        matched = any(
            s.bank == expected.bank and any(s.covers(p) for p in expected.pages)
            for s in response.sources
        )
        if not matched:
            cited = ", ".join(s.label for s in response.sources) or "nothing"
            failures.append(f"no {expected.bank.value} citation covering page(s) {expected.pages}; cited: {cited}")
    return failures


def normalize_ws(text: str) -> str:
    """Collapse every run of whitespace to one space, so a phrase split by a PDF line wrap still matches."""
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class VariantResult:
    """The outcome of asking one phrasing of a question."""

    question: str
    failures: list[str]
    answer: str = ""
    error: str | None = None  # the provider failed (rate limit, timeout...): correctness was not evaluated

    @property
    def passed(self) -> bool:
        """True if the answer met every expectation and the provider did not fail."""
        return not self.failures and self.error is None


@dataclass
class CaseResult:
    """The outcomes of every phrasing of one case."""

    case_id: str
    variants: list[VariantResult] = field(default_factory=list)
    known_issue: str | None = None

    @property
    def passed(self) -> bool:
        """True only if every phrasing passed."""
        return all(v.passed for v in self.variants)

    @property
    def status(self) -> str:
        """PASS, FAIL (a wrong answer was seen) or ERROR (only provider errors, nothing wrong seen).

        A case marked with a known issue reports KNOWN instead of FAIL, and FIXED instead of PASS.
        """
        if self.passed:
            return "FIXED" if self.known_issue else "PASS"
        if any(v.failures for v in self.variants):
            return "KNOWN" if self.known_issue else "FAIL"
        return "ERROR"

    @property
    def ok(self) -> bool:
        """False only for a genuine failure or a provider error; known issues do not fail a run."""
        return self.status in {"PASS", "FIXED", "KNOWN"}


async def run_eval(service: ChatService, cases: list[EvalCase], delay_s: float = 0.0) -> list[CaseResult]:
    """Ask every variant of every case, one at a time, each in its own fresh session.

    A case with `context` first asks those earlier turns in the same session (not judged), so its
    question is a real follow-up. `delay_s` pauses before every question after the first, to stay
    under the provider's rate limit.
    """
    results: list[CaseResult] = []
    first = True
    for case in cases:
        result = CaseResult(case.id, known_issue=case.known_issue)
        for n, question in enumerate(case.variants):
            session_id = f"eval-{case.id}-{n}"[:64]
            error: str | None = None
            response: ChatResponse | None = None
            for asked in [*case.context, question]:
                if not first and delay_s:
                    await asyncio.sleep(delay_s)
                first = False
                try:
                    response = await service.ask(ChatRequest(session_id=session_id, question=asked))
                except ChatServiceError as exc:
                    error = str(exc)
                    break  # a failed setup turn makes the judged answer meaningless
            if error is not None or response is None:
                result.variants.append(VariantResult(question, [], error=error or "no answer"))
            else:
                result.variants.append(VariantResult(question, check_response(case, response), response.answer))
        results.append(result)
    return results


@dataclass
class RetrievalResult:
    """The outcome of the retrieval-only check for one phrasing of a question."""

    case_id: str
    question: str
    found: bool  # some retrieved passage holds all the evidence strings, for the right bank
    pages_ok: bool  # ...and that passage covers an expected page
    retrieved: list[str]  # citation labels, best first
    known_issue: str | None = None

    @property
    def passed(self) -> bool:
        """True if the evidence was retrieved from an expected page."""
        return self.found and self.pages_ok

    @property
    def status(self) -> str:
        """PASS or FAIL; KNOWN or FIXED when the case carries a documented known issue."""
        if self.passed:
            return "FIXED" if self.known_issue else "PASS"
        return "KNOWN" if self.known_issue else "FAIL"

    @property
    def ok(self) -> bool:
        """False only for a genuine failure; known issues do not fail a run."""
        return self.status != "FAIL"


def evaluate_retrieval(retriever: BaseRetriever, cases: list[EvalCase]) -> list[RetrievalResult]:
    """Check that retrieval alone surfaces each verified fact, for every phrasing. No LLM needed.

    If retrieval never surfaces the passage holding the answer, no model can answer correctly,
    so this isolates retrieval problems from model problems. A case with several evidence items
    (a comparison across banks) passes only if every one is retrieved. Follow-up cases are skipped:
    turning a follow-up into a standalone question takes an LLM.
    """
    results: list[RetrievalResult] = []
    for case in cases:
        if not case.evidence or case.context:
            continue
        for question in case.variants:
            docs = retriever.invoke(question)
            found_all, pages_all = True, True
            for evidence in case.evidence:
                matches = [
                    d
                    for d in docs
                    if d.metadata.get("bank") == evidence.bank.value
                    and all(normalize_ws(term) in normalize_ws(d.page_content) for term in evidence.contains)
                ]
                citations = [ChunkMetadata.model_validate(d.metadata).to_citation() for d in matches]
                wanted = [s for s in case.expect_sources if s.bank == evidence.bank]
                found_all &= bool(matches)
                pages_all &= not wanted or any(c.covers(p) for c in citations for s in wanted for p in s.pages)
            labels = [ChunkMetadata.model_validate(d.metadata).to_citation().label for d in docs]
            results.append(RetrievalResult(case.id, question, found_all, pages_all, labels, case.known_issue))
    return results


def format_retrieval_report(results: list[RetrievalResult]) -> str:
    """One line per question; retrieved passages are listed for the ones that failed."""
    lines = []
    for r in results:
        lines.append(f"{r.status:5} {r.case_id:34} {r.question[:60]}")
        if not r.passed:
            reason = "evidence not in any retrieved passage" if not r.found else "found, but not on an expected page"
            lines.append(f"      {reason}; retrieved: {'; '.join(r.retrieved) or 'nothing'}")
        if r.status == "KNOWN":
            lines.append(f"      known issue: {r.known_issue}")
        elif r.status == "FIXED":
            lines.append("      now passing: remove known_issue from this case in the golden set")
    known = sum(r.status == "KNOWN" for r in results)
    summary = f"\nretrieval recall: {sum(r.passed for r in results)}/{len(results)} questions"
    if known:
        summary += f" ({known} known issue{'s' if known != 1 else ''}, see above)"
    lines.append(summary)
    return "\n".join(lines)


def format_report(results: list[CaseResult]) -> str:
    """Plain-text report: one line per case, details for every failing variant."""
    lines: list[str] = []
    for r in results:
        passed = sum(v.passed for v in r.variants)
        lines.append(f"{r.status:5} {r.case_id}  ({passed}/{len(r.variants)} variants)")
        if r.status == "KNOWN":
            lines.append(f"      known issue: {r.known_issue}")
        elif r.status == "FIXED":
            lines.append("      now passing: remove known_issue from this case in the golden set")
        for v in r.variants:
            if v.error is not None:
                lines.append(f"      ! not evaluated: {v.question}")
                lines.append(f"        provider error: {v.error}")
            elif v.failures:
                lines.append(f"      ? {v.question}")
                lines.extend(f"        - {failure}" for failure in v.failures)
    total = sum(len(r.variants) for r in results)
    ok = sum(v.passed for r in results for v in r.variants)
    errored = sum(v.error is not None for r in results for v in r.variants)
    summary = f"\n{sum(r.passed for r in results)}/{len(results)} cases passed, {ok}/{total} questions passed"
    known = sum(r.status == "KNOWN" for r in results)
    if known:
        summary += f"\n{known} known issue{'s' if known != 1 else ''} (documented, not counted as failures)"
    if errored:
        summary += f"\n{errored} question(s) were NOT evaluated because the provider failed; re-run them (see --case, --delay)"
    lines.append(summary)
    return "\n".join(lines)
