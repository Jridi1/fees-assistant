"""Validated data models shared by the API, the RAG pipeline and the clients."""

from enum import Enum
from pathlib import PurePath
from typing import Annotated, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

MAX_QUESTION_LENGTH = 500

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_LENGTH),
]
SessionId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class Bank(str, Enum):
    """Banks whose documents are indexed. Add a member to support a new bank."""

    N26 = "N26"
    REVOLUT = "Revolut"


class ChunkMetadata(BaseModel):
    """Metadata every indexed chunk must carry, validated at ingestion time.

    Mirrors what PyMuPDFLoader produces, plus the `bank` tag added by the loader.
    Unknown keys are ignored. `page` is 0-indexed, as in the loader.
    """

    model_config = ConfigDict(extra="ignore")

    bank: Bank
    source: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=0)
    page_end: int | None = Field(default=None, ge=0)  # last page when a chunk spans a page break
    chunk_index: int | None = Field(default=None, ge=0)  # position within the source file

    @model_validator(mode="after")
    def _page_range_is_ordered(self) -> "ChunkMetadata":
        if self.page_end is not None and (self.page is None or self.page_end < self.page):
            raise ValueError("page_end must not be before page")
        return self

    def to_citation(self) -> "SourceCitation":
        """Convert to a user-facing citation (file name only, 1-indexed pages)."""
        spans = self.page is not None and self.page_end is not None and self.page_end > self.page
        return SourceCitation(
            bank=self.bank,
            document=PurePath(self.source.replace("\\", "/")).name,
            page=None if self.page is None else self.page + 1,
            page_end=self.page_end + 1 if spans else None,
        )


class SourceCitation(BaseModel):
    """Where an answer came from. Frozen so it is hashable and easy to de-duplicate."""

    model_config = ConfigDict(frozen=True)

    bank: Bank
    document: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)  # set only when the passage spans pages

    @model_validator(mode="after")
    def _page_range_is_ordered(self) -> "SourceCitation":
        if self.page_end is not None and (self.page is None or self.page_end <= self.page):
            raise ValueError("page_end must be after page")
        return self

    def covers(self, page: int) -> bool:
        """True if this citation includes the given 1-indexed page."""
        if self.page is None:
            return False
        return self.page <= page <= (self.page_end or self.page)

    @property
    def label(self) -> str:
        """Human-readable form, e.g. 'N26 — 13account-pricelist-en.pdf, page 6' or 'pages 5–6'."""
        if self.page is None:
            pages = ""
        elif self.page_end is not None:
            pages = f", pages {self.page}–{self.page_end}"
        else:
            pages = f", page {self.page}"
        return f"{self.bank.value} — {self.document}{pages}"


def unique_citations(citations: Iterable[SourceCitation]) -> list[SourceCitation]:
    """Drop duplicate citations, keeping first-seen order."""
    return list(dict.fromkeys(citations))


class LLMAnswer(BaseModel):
    """Structured output the LLM must return; validated before it reaches the user.

    The field descriptions are sent to the model as part of the schema.
    """

    answer: str = Field(min_length=1, description="The answer in markdown, or a short statement that the documents do not cover the question.")
    answer_found: bool = Field(description="True only if the numbered context passages contain the information needed to answer. False if the answer is not in them.")
    source_ids: list[int] = Field(default_factory=list, description="Numbers of the context passages actually used for the answer. Empty when answer_found is false.")


class ChatRequest(BaseModel):
    """A user question within a conversation session."""

    session_id: SessionId
    question: Question


class ChatResponse(BaseModel):
    """The assistant's answer.

    `grounded` is False when the assistant declined because the documents do not
    contain the answer. A declined answer must not carry citations.
    """

    session_id: SessionId
    answer: str = Field(min_length=1)
    grounded: bool
    sources: list[SourceCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refusals_have_no_sources(self) -> "ChatResponse":
        if not self.grounded and self.sources:
            raise ValueError("an ungrounded (refused) answer must not cite sources")
        return self


class ChatEvent(BaseModel):
    """One step of a streamed answer: progress updates, then the final answer (or an error)."""

    type: Literal["status", "answer", "error"]
    message: str | None = None  # status and error
    code: str | None = None  # error only: rate_limited, timeout or unavailable
    response: ChatResponse | None = None  # answer only

    @model_validator(mode="after")
    def _fields_match_type(self) -> "ChatEvent":
        if self.type == "answer" and self.response is None:
            raise ValueError("an 'answer' event needs a response")
        if self.type != "answer" and (self.response is not None or not self.message):
            raise ValueError("'status' and 'error' events need a message and no response")
        return self

    @classmethod
    def status(cls, message: str) -> "ChatEvent":
        """A progress update, such as "Searching the documents"."""
        return cls(type="status", message=message)

    @classmethod
    def answer(cls, response: ChatResponse) -> "ChatEvent":
        """The final, validated answer. It ends a successful stream."""
        return cls(type="answer", response=response)

    @classmethod
    def error(cls, code: str, message: str) -> "ChatEvent":
        """A failure that ends the stream. `code` is machine-readable (rate_limited, timeout, unavailable)."""
        return cls(type="error", code=code, message=message)
