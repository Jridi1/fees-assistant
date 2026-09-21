"""Async chat orchestration: condense follow-ups, retrieve, answer with citations.

Depends only on `langchain-core`. The retriever and the two chains are injected,
so this module can be tested without a vector database or a real LLM.
"""

import asyncio
import logging
from typing import Any, AsyncIterator, Protocol, Sequence

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from feesbot.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, NO_DOCUMENTS_ANSWER
from feesbot.schemas import (
    ChatEvent,
    ChatRequest,
    ChatResponse,
    ChunkMetadata,
    LLMAnswer,
    SourceCitation,
    unique_citations,
)
from feesbot.sessions import SessionStore, Turn
from feesbot.settings import Settings

logger = logging.getLogger(__name__)

MAX_STRUCTURED_ATTEMPTS = 2


class ChatServiceError(Exception):
    """The assistant could not produce an answer (retrieval or LLM failure)."""


class ChatTimeoutError(ChatServiceError):
    """The request exceeded `request_timeout_s`."""


class ChatRateLimitError(ChatServiceError):
    """The LLM provider is rate-limiting requests; retrying later may succeed."""


class Retriever(Protocol):
    """Anything that can asynchronously fetch relevant documents for a query."""

    async def ainvoke(self, query: str) -> Sequence[Any]:
        """Return the passages relevant to `query`, best first. Each has `page_content` and `metadata`."""
        ...


def build_condense_chain(llm: BaseChatModel) -> Runnable:
    """Chain that rewrites a follow-up question as a standalone one (returns a string)."""
    return CONDENSE_PROMPT | llm | StrOutputParser()


def build_answer_chain(llm: BaseChatModel) -> Runnable:
    """Chain that answers from numbered context and returns a validated `LLMAnswer`."""
    return ANSWER_PROMPT | llm.with_structured_output(LLMAnswer)


def format_history(turns: Sequence[Turn]) -> str:
    """Render past turns as plain text for the condense prompt."""
    return "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in turns)


def format_context(docs: Sequence[Any]) -> tuple[str, list[SourceCitation]]:
    """Number the retrieved passages and label each with its bank, file and page.

    Returns the prompt context and the citations, aligned so that passage `n`
    corresponds to `citations[n - 1]`.

    Raises:
        ValidationError: if a document lacks valid metadata (a bug in ingestion).
    """
    citations = [ChunkMetadata.model_validate(d.metadata).to_citation() for d in docs]
    blocks = [
        f"[{i}] {citation.label}\n{doc.page_content}"
        for i, (doc, citation) in enumerate(zip(docs, citations), start=1)
    ]
    return "\n\n".join(blocks), citations


def select_citations(result: LLMAnswer, citations: Sequence[SourceCitation]) -> list[SourceCitation]:
    """Choose which citations to show.

    A refusal cites nothing. Otherwise cite the passages the model says it used,
    ignoring out-of-range numbers; if none are valid, fall back to everything retrieved.
    """
    if not result.answer_found:
        return []
    used = [i for i in dict.fromkeys(result.source_ids) if 1 <= i <= len(citations)]
    if not used:
        logger.warning("Model gave no valid source_ids %s; citing all retrieved passages", result.source_ids)
        return unique_citations(citations)
    return unique_citations(citations[i - 1] for i in used)


class ChatService:
    """Answers questions from indexed documents, with per-session memory.

    One instance is shared by all requests. Requests within the same session are
    processed one at a time; different sessions run concurrently.
    """

    def __init__(
        self,
        retriever: Retriever,
        condense_chain: Runnable,
        answer_chain: Runnable,
        sessions: SessionStore,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._condense_chain = condense_chain
        self._answer_chain = answer_chain
        self._sessions = sessions
        self._timeout_s = settings.request_timeout_s

    async def ask(self, request: ChatRequest) -> ChatResponse:
        """Answer one question and record it in the session history.

        Raises:
            ChatTimeoutError: if the request takes longer than the configured timeout.
            ChatRateLimitError: if the LLM provider is rate-limiting.
            ChatServiceError: on any other retrieval or LLM failure.
        """
        response: ChatResponse | None = None
        async for event in self.ask_stream(request):  # consume fully so the turn is recorded
            if event.response is not None:
                response = event.response
        if response is None:
            raise ChatServiceError("The assistant produced no answer.")
        return response

    async def ask_stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        """Answer one question as a series of events: status updates, then the answer.

        The turn is recorded in the session history only if the answer completes; a caller
        that stops iterating early (e.g. the client disconnected) leaves the history untouched.
        Raises the same errors as `ask`.
        """
        session_id = request.session_id
        async with self._sessions.lock(session_id):
            response: ChatResponse | None = None
            try:
                async with asyncio.timeout(self._timeout_s):
                    async for event in self._events(request):
                        if event.response is not None:
                            response = event.response
                        yield event
            except TimeoutError as exc:
                raise ChatTimeoutError(f"No answer within {self._timeout_s:g}s") from exc
            except ChatServiceError:
                raise
            except Exception as exc:  # service boundary: callers get one error type
                if type(exc).__name__ == "RateLimitError":  # groq/openai SDKs; matched by name to avoid importing them here
                    logger.warning("LLM provider rate limit hit (session %s)", session_id)
                    raise ChatRateLimitError("The LLM provider is rate-limiting requests.") from exc
                logger.exception("Chat request failed (session %s)", session_id)
                raise ChatServiceError("The assistant could not answer right now.") from exc
            if response is not None:
                self._sessions.add_turn(session_id, request.question, response.answer)

    async def _events(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        session_id = request.session_id
        history = self._sessions.history(session_id)
        if history:
            yield ChatEvent.status("Reading the conversation so far...")
        question = await self._standalone_question(history, request.question)

        yield ChatEvent.status("Searching the documents...")
        docs = await self._retriever.ainvoke(question)
        if not docs:
            yield ChatEvent.answer(ChatResponse(session_id=session_id, answer=NO_DOCUMENTS_ANSWER, grounded=False))
            return

        context, citations = format_context(docs)
        yield ChatEvent.status("Writing the answer...")
        result = await self._structured_answer(context, question)
        yield ChatEvent.answer(
            ChatResponse(
                session_id=session_id,
                answer=result.answer,
                grounded=result.answer_found,
                sources=select_citations(result, citations),
            )
        )

    async def _standalone_question(self, history: Sequence[Turn], question: str) -> str:
        if not history:
            return question
        rewritten = await self._condense_chain.ainvoke(
            {"history": format_history(history), "question": question}
        )
        return rewritten.strip() or question

    async def _structured_answer(self, context: str, question: str) -> LLMAnswer:
        for attempt in range(1, MAX_STRUCTURED_ATTEMPTS + 1):
            try:
                result = await self._answer_chain.ainvoke({"context": context, "question": question})
                if result is None:
                    raise OutputParserException("model returned no structured output")
                return result if isinstance(result, LLMAnswer) else LLMAnswer.model_validate(result)
            except (OutputParserException, ValidationError) as exc:
                logger.warning("Invalid structured answer (attempt %d/%d): %s", attempt, MAX_STRUCTURED_ATTEMPTS, exc)
        raise ChatServiceError("The model did not return a valid answer.")
