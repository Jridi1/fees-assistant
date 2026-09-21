import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from feesbot.chat import ChatRateLimitError, ChatService, ChatServiceError, ChatTimeoutError, select_citations
from feesbot.prompts import NO_DOCUMENTS_ANSWER
from feesbot.schemas import ChatRequest, LLMAnswer, SourceCitation
from feesbot.sessions import SessionStore
from feesbot.settings import Settings


def doc(bank, filename, page, text):
    return Document(page_content=text, metadata={"bank": bank, "source": f"{bank}/{filename}", "page": page})


DOCS = [
    doc("N26", "13account-pricelist-en.pdf", 5, "Replacement card: EUR 45"),
    doc("Revolut", "internal_policy.pdf", 2, "Replacement card: EUR 6"),
]


class FakeRetriever:
    def __init__(self, docs, delay=0.0, error=None):
        self.docs, self.delay, self.error, self.queries = docs, delay, error, []

    async def ainvoke(self, query):
        self.queries.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.docs


class Recorder:
    """A runnable that records its inputs and returns scripted outputs."""

    def __init__(self, *outputs):
        self.outputs, self.inputs = list(outputs), []

    def as_runnable(self):
        async def call(value):
            self.inputs.append(value)
            out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
            if isinstance(out, Exception):
                raise out
            return out

        return RunnableLambda(lambda v: None, afunc=call)


def make_service(retriever, answers, condensed="standalone question", timeout=5.0):
    settings = Settings(_env_file=None, groq_api_key="x", request_timeout_s=timeout)
    condense, answer = Recorder(condensed), Recorder(*answers)
    service = ChatService(
        retriever=retriever,
        condense_chain=condense.as_runnable(),
        answer_chain=answer.as_runnable(),
        sessions=SessionStore(settings.max_sessions, settings.max_history_turns),
        settings=settings,
    )
    return service, condense, answer


def ask(service, question="How much is a new card?", session="s1"):
    return asyncio.run(service.ask(ChatRequest(session_id=session, question=question)))


GOOD = LLMAnswer(answer="N26 charges EUR 45.", answer_found=True, source_ids=[1])


def test_grounded_answer_cites_only_the_passages_used():
    service, _, _ = make_service(FakeRetriever(DOCS), [GOOD])
    response = ask(service)
    assert response.grounded
    assert [s.label for s in response.sources] == ["N26 — 13account-pricelist-en.pdf, page 6"]


def test_refusal_is_ungrounded_and_cites_nothing():
    refusal = LLMAnswer(answer="Not in the documents.", answer_found=False, source_ids=[1, 2])
    service, _, _ = make_service(FakeRetriever(DOCS), [refusal])
    response = ask(service)
    assert not response.grounded and response.sources == []


def test_invalid_source_ids_fall_back_to_everything_retrieved():
    sloppy = LLMAnswer(answer="Both charge.", answer_found=True, source_ids=[9])
    service, _, _ = make_service(FakeRetriever(DOCS), [sloppy])
    assert len(ask(service).sources) == 2


def test_no_retrieved_documents_short_circuits_without_calling_the_llm():
    service, _, answer = make_service(FakeRetriever([]), [GOOD])
    response = ask(service)
    assert response.answer == NO_DOCUMENTS_ANSWER and not response.grounded
    assert answer.inputs == []


def test_context_sent_to_llm_is_numbered_and_labelled():
    service, _, answer = make_service(FakeRetriever(DOCS), [GOOD])
    ask(service)
    context = answer.inputs[0]["context"]
    assert context.startswith("[1] N26 — 13account-pricelist-en.pdf, page 6\nReplacement card: EUR 45")
    assert "[2] Revolut — internal_policy.pdf, page 3" in context


def test_follow_up_is_rewritten_using_history_but_first_question_is_not():
    retriever = FakeRetriever(DOCS)
    service, condense, _ = make_service(retriever, [GOOD], condensed="What does Revolut charge for a card?")
    ask(service, "How much is a new card?")
    assert condense.inputs == [] and retriever.queries == ["How much is a new card?"]
    ask(service, "and Revolut?")
    assert "N26 charges EUR 45." in condense.inputs[0]["history"]
    assert retriever.queries[-1] == "What does Revolut charge for a card?"


def test_sessions_do_not_share_history():
    service, condense, _ = make_service(FakeRetriever(DOCS), [GOOD])
    ask(service, "first", session="alice")
    ask(service, "hello from bob", session="bob")
    assert condense.inputs == []  # bob's first message must not see alice's history


def test_one_invalid_answer_is_retried():
    service, _, answer = make_service(FakeRetriever(DOCS), [OutputParserException("bad json"), GOOD])
    assert ask(service).grounded
    assert len(answer.inputs) == 2


def test_repeated_invalid_answers_raise_a_service_error():
    service, _, _ = make_service(FakeRetriever(DOCS), [OutputParserException("bad json")])
    with pytest.raises(ChatServiceError):
        ask(service)


def test_retriever_failure_becomes_a_service_error_and_is_not_remembered():
    service, _, _ = make_service(FakeRetriever(DOCS, error=RuntimeError("db down")), [GOOD])
    with pytest.raises(ChatServiceError):
        ask(service)
    assert service._sessions.history("s1") == []


def test_a_provider_rate_limit_is_reported_as_such():
    class RateLimitError(Exception):  # same name as the groq / openai SDK exception
        pass

    service, _, _ = make_service(FakeRetriever(DOCS, error=RateLimitError("429")), [GOOD])
    with pytest.raises(ChatRateLimitError):
        ask(service)
    assert issubclass(ChatRateLimitError, ChatServiceError)  # callers catching the base class still work


def test_slow_request_times_out():
    service, _, _ = make_service(FakeRetriever(DOCS, delay=1.0), [GOOD], timeout=0.05)
    with pytest.raises(ChatTimeoutError):
        ask(service)


def test_same_session_requests_are_processed_one_at_a_time():
    async def scenario():
        service, condense, _ = make_service(FakeRetriever(DOCS, delay=0.05), [GOOD])
        req = lambda q: ChatRequest(session_id="s1", question=q)
        await asyncio.gather(service.ask(req("one")), service.ask(req("two")))
        return condense.inputs

    # the second request only starts after the first is recorded, so it sees 1 turn of history
    inputs = asyncio.run(scenario())
    assert len(inputs) == 1 and "User: one" in inputs[0]["history"]


def collect(service, question="How much is a new card?", session="s1"):
    async def run():
        return [e async for e in service.ask_stream(ChatRequest(session_id=session, question=question))]

    return asyncio.run(run())


def test_stream_reports_progress_then_the_answer():
    service, _, _ = make_service(FakeRetriever(DOCS), [GOOD])
    events = collect(service)
    assert [(e.type, e.message) for e in events[:-1]] == [
        ("status", "Searching the documents..."),
        ("status", "Writing the answer..."),
    ]
    assert events[-1].type == "answer" and events[-1].response.grounded


def test_stream_mentions_reading_the_conversation_only_for_follow_ups():
    service, _, _ = make_service(FakeRetriever(DOCS), [GOOD])
    collect(service)
    follow_up = collect(service, "and Revolut?")
    assert follow_up[0].message == "Reading the conversation so far..."


def test_stream_with_no_documents_ends_with_a_refusal():
    service, _, _ = make_service(FakeRetriever([]), [GOOD])
    last = collect(service)[-1]
    assert last.type == "answer" and not last.response.grounded and last.response.answer == NO_DOCUMENTS_ANSWER


def test_abandoning_a_stream_records_nothing_and_frees_the_session():
    async def scenario():
        service, _, _ = make_service(FakeRetriever(DOCS), [GOOD])
        stream = service.ask_stream(ChatRequest(session_id="s1", question="q"))
        await stream.__anext__()  # the client reads one event, then disconnects
        await stream.aclose()
        recorded = service._sessions.history("s1")
        # the session lock must be free again, otherwise this would hang
        response = await asyncio.wait_for(service.ask(ChatRequest(session_id="s1", question="q2")), timeout=2)
        return recorded, response

    recorded, response = asyncio.run(scenario())
    assert recorded == [] and response.grounded


def test_select_citations_ignores_duplicates():
    c = SourceCitation(bank="N26", document="a.pdf", page=1)
    result = LLMAnswer(answer="x", answer_found=True, source_ids=[1, 1, 2])
    assert select_citations(result, [c, c]) == [c]
