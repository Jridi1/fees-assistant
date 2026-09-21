"""Application settings, loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from feesbot.schemas import Bank


class MissingSecretError(RuntimeError):
    """A secret needed to run the service is not configured."""


class Settings(BaseSettings):
    """Typed configuration. Fails at startup if a required value is missing or invalid."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # secrets (env vars GROQ_API_KEY, DC_API_KEY). Optional here so that building the index does not
    # need them (e.g. while building a Docker image); serving calls `require_groq_key()`.
    groq_api_key: SecretStr | None = None
    dc_api_key: SecretStr | None = None  # only needed for the Discord bot

    # LLM
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = Field(default=0.5, ge=0.0, le=2.0)

    # retrieval
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chroma_dir: Path = Path("chroma_db")
    # Hybrid (embeddings + BM25) search over 600-char chunks, each returned together with its
    # neighbours so a price-list row (heading, description, price) reaches the model whole.
    # Chosen by measuring evidence recall on the golden set: dense-only retrieval never
    # surfaced the N26 standard replacement fee; this setup finds all 15 test questions.
    retrieval_k: int = Field(default=4, ge=1, le=20)
    retrieval_candidates: int = Field(default=30, ge=1, le=200)  # per method, before fusion
    neighbor_window: int = Field(default=1, ge=0, le=3)  # chunks added on each side of a hit
    use_multi_query: bool = False  # extra LLM call per question; hybrid search already handles paraphrases
    chunk_size: int = Field(default=600, ge=100, le=4000)
    chunk_overlap: int = Field(default=90, ge=0)

    # HTTP API: browser origins allowed to call it, e.g. CORS_ORIGINS='["https://you.pages.dev"]'
    cors_origins: list[str] = Field(default_factory=list)

    # abuse limits (per process). Every question costs LLM tokens, so a public demo needs caps.
    rate_limit_per_minute: int = Field(default=8, ge=1)  # per visitor
    global_rate_limit_per_minute: int = Field(default=20, ge=1)  # all visitors together
    max_concurrent_per_client: int = Field(default=2, ge=1)
    max_concurrent_total: int = Field(default=6, ge=1)
    daily_request_budget: int = Field(default=300, ge=1)  # resets at 00:00 UTC
    # How many reverse proxies sit in front of the app. 0 = use the socket address and ignore
    # X-Forwarded-For (safe default). Behind a proxy set this to the number of proxies: too low and
    # every visitor shares one limit; too high and clients can spoof their address.
    trusted_proxy_hops: int = Field(default=0, ge=0, le=5)

    # conversation
    max_sessions: int = Field(default=1000, ge=1)  # oldest idle sessions are evicted
    max_history_turns: int = Field(default=10, ge=1, le=50)
    request_timeout_s: float = Field(default=60.0, gt=0)

    # bank -> folder of PDFs, or a single PDF
    bank_sources: dict[Bank, Path] = {
        Bank.N26: Path("N26"),
        Bank.REVOLUT: Path("internal_policy.pdf"),
    }

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def require_groq_key(self) -> SecretStr:
        """The Groq key, or a clear error. Call before doing anything expensive when serving."""
        if self.groq_api_key is None or not self.groq_api_key.get_secret_value().strip():
            raise MissingSecretError("GROQ_API_KEY is not set. Put it in .env or the environment.")
        return self.groq_api_key


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse the same instance."""
    return Settings()
