"""HTTP API. Run with: python -m feesbot serve   (or: uvicorn feesbot.api:create_app --factory)

    GET  /              the chat page (add ?embed=1 when it is shown inside another site)
    GET  /health        liveness check
    POST /chat          question in, complete answer out
    POST /chat/stream   same, as Server-Sent Events: `status` updates, then `answer` (or `error`)
"""

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.background import BackgroundTask

from feesbot.chat import ChatRateLimitError, ChatService, ChatServiceError, ChatTimeoutError
from feesbot.limits import Lease, LimitExceeded, RateLimiter, client_ip
from feesbot.schemas import ChatEvent, ChatRequest, ChatResponse
from feesbot.settings import Settings, get_settings

logger = logging.getLogger(__name__)

RETRY_AFTER_SECONDS = 10
STATIC_DIR = Path(__file__).parent / "static"
_ORIGIN = re.compile(r"^https?://[^\s;,'\"]+$")


def page_headers(origins: list[str]) -> dict[str, str]:
    """Security headers for the chat page.

    The page may only load its own scripts and call its own API. `frame-ancestors` decides who may
    embed it in an iframe: itself and the configured portfolio origins, nobody else.
    """
    allowed = []
    for origin in origins:
        if _ORIGIN.match(origin):
            allowed.append(origin.rstrip("/"))
        else:
            logger.warning("Ignoring malformed origin for frame-ancestors: %r", origin)
    csp = "; ".join(
        [
            "default-src 'none'",
            "script-src 'self'",
            "style-src 'self' https://fonts.googleapis.com",
            "font-src https://fonts.gstatic.com",
            "img-src 'self' data:",
            "connect-src 'self'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors " + " ".join(["'self'", *allowed]),
        ]
    )
    return {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
    }


class ErrorResponse(BaseModel):
    """Body of every error response."""

    # rate_limited (the LLM provider), timeout, unavailable, or from our own limits:
    # too_many_requests (this visitor), busy (everyone), budget_exhausted (today's budget)
    code: str
    detail: str


def _classify(exc: ChatServiceError) -> tuple[int, str]:
    """HTTP status and machine-readable code for a service error."""
    if isinstance(exc, ChatRateLimitError):
        return 429, "rate_limited"
    if isinstance(exc, ChatTimeoutError):
        return 504, "timeout"
    return 503, "unavailable"


def format_sse(event: ChatEvent) -> str:
    """One Server-Sent Event frame."""
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def get_service(request: Request) -> ChatService:
    """FastAPI dependency: the shared ChatService."""
    return request.app.state.service


def build_limiter(settings: Settings) -> RateLimiter:
    """Limiter configured from settings. A lease outliving the request timeout is treated as leaked."""
    return RateLimiter(
        per_client_per_minute=settings.rate_limit_per_minute,
        global_per_minute=settings.global_rate_limit_per_minute,
        max_concurrent_per_client=settings.max_concurrent_per_client,
        max_concurrent_total=settings.max_concurrent_total,
        daily_budget=settings.daily_request_budget,
        lease_ttl_s=settings.request_timeout_s + 30,
    )


async def _sse_stream(service: ChatService, body: ChatRequest, lease: Lease | None = None) -> AsyncIterator[str]:
    try:
        async for event in service.ask_stream(body):
            yield format_sse(event)
    except ChatServiceError as exc:  # the HTTP status is already sent, so report the error as an event
        _, code = _classify(exc)
        yield format_sse(ChatEvent.error(code, str(exc)))
    finally:
        if lease is not None:
            lease.release()


def create_app(
    service: ChatService | None = None,
    settings: Settings | None = None,
    limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the app. Without `service`, the index and LLM are loaded at startup.

    Passing a `service` (and `settings`, and optionally a `limiter`) is how the tests run the API
    without any model.
    """
    settings = settings or get_settings()
    if service is None:
        settings.require_groq_key()  # a server that cannot answer should not start at all
    limiter = limiter or build_limiter(settings)

    def admit(request: Request) -> Lease:
        """Apply the abuse limits to a chat request; raises LimitExceeded (answered as HTTP 429/503)."""
        visitor = client_ip(
            request.client.host if request.client else None,
            request.headers.get("x-forwarded-for"),
            settings.trusted_proxy_hops,
        )
        return limiter.acquire(visitor)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            from feesbot.bootstrap import create_chat_service  # heavy imports: only when really serving

            logger.info("Loading index and models...")
            app.state.service = await asyncio.to_thread(create_chat_service, settings)
            logger.info("Ready.")
        yield

    app = FastAPI(title="Multi-bank fees assistant", lifespan=lifespan)
    app.state.service = service
    app.state.limiter = limiter

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(ChatServiceError)
    async def service_error_handler(_: Request, exc: ChatServiceError) -> JSONResponse:
        status, code = _classify(exc)
        headers = {"Retry-After": str(RETRY_AFTER_SECONDS)} if status == 429 else None
        return JSONResponse(ErrorResponse(code=code, detail=str(exc)).model_dump(), status_code=status, headers=headers)

    @app.exception_handler(LimitExceeded)
    async def limit_handler(_: Request, exc: LimitExceeded) -> JSONResponse:
        status = 503 if exc.code == "busy" else 429
        return JSONResponse(
            ErrorResponse(code=exc.code, detail=exc.detail).model_dump(),
            status_code=status,
            headers={"Retry-After": str(exc.retry_after)},
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    headers = page_headers(settings.cors_origins)

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers=headers)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.post("/chat", response_model=ChatResponse, responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 504: {"model": ErrorResponse}})
    async def chat(request: Request, body: ChatRequest, service: ChatService = Depends(get_service)) -> ChatResponse:
        lease = admit(request)
        try:
            return await service.ask(body)
        finally:
            lease.release()

    @app.post("/chat/stream", responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    async def chat_stream(request: Request, body: ChatRequest, service: ChatService = Depends(get_service)) -> StreamingResponse:
        lease = admit(request)  # refused requests never start a stream
        return StreamingResponse(
            _sse_stream(service, body, lease),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            background=BackgroundTask(lease.release),  # also release if the stream generator never ran (idempotent)
        )

    return app
