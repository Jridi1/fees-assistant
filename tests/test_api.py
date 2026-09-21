import json

import pytest
from fastapi.testclient import TestClient

from feesbot.api import create_app, format_sse
from feesbot.chat import ChatRateLimitError, ChatServiceError, ChatTimeoutError
from feesbot.limits import RateLimiter
from feesbot.schemas import ChatEvent, ChatResponse, SourceCitation
from feesbot.settings import MissingSecretError, Settings
from test_chat import DOCS, GOOD, FakeRetriever, make_service

ORIGIN = "https://me.pages.dev"
SETTINGS = Settings(_env_file=None, groq_api_key="x", cors_origins=[ORIGIN])
CITATION = SourceCitation(bank="N26", document="13account-pricelist-en.pdf", page=5, page_end=6)
ANSWER = ChatResponse(session_id="s1", answer="N26 charges 10.00 EUR.", grounded=True, sources=[CITATION])
BODY = {"session_id": "s1", "question": "How much is a replacement card?"}


class ScriptedService:
    """Stands in for ChatService: returns ANSWER, or raises `error`."""

    def __init__(self, error=None, raise_after_first_event=False):
        self.error = error
        self.raise_after_first_event = raise_after_first_event

    async def ask(self, request):
        if self.error:
            raise self.error
        return ANSWER

    async def ask_stream(self, request):
        yield ChatEvent.status("Searching the documents...")
        if self.error:
            raise self.error
        yield ChatEvent.answer(ANSWER)


def client_for(service):
    return TestClient(create_app(service=service, settings=SETTINGS))


def parse_sse(text):
    events = []
    for block in filter(None, text.split("\n\n")):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_health():
    assert client_for(ScriptedService()).get("/health").json() == {"status": "ok"}


def test_chat_returns_the_validated_answer_with_citations():
    response = client_for(ScriptedService()).post("/chat", json=BODY)
    assert response.status_code == 200
    data = response.json()
    assert data["grounded"] is True and data["answer"] == "N26 charges 10.00 EUR."
    assert data["sources"] == [{"bank": "N26", "document": "13account-pricelist-en.pdf", "page": 5, "page_end": 6}]


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "s1", "question": "   "},
        {"session_id": "s1", "question": "x" * 501},
        {"session_id": "../etc/passwd", "question": "hi"},
        {"question": "no session"},
    ],
)
def test_invalid_requests_are_rejected_before_reaching_the_service(payload):
    assert client_for(ScriptedService(error=AssertionError("must not be called"))).post("/chat", json=payload).status_code == 422


@pytest.mark.parametrize(
    "error, status, code",
    [
        (ChatRateLimitError("slow down"), 429, "rate_limited"),
        (ChatTimeoutError("too slow"), 504, "timeout"),
        (ChatServiceError("broken"), 503, "unavailable"),
    ],
)
def test_service_errors_map_to_http_statuses(error, status, code):
    response = client_for(ScriptedService(error=error)).post("/chat", json=BODY)
    assert response.status_code == status
    assert response.json() == {"code": code, "detail": str(error)}
    assert ("retry-after" in response.headers) == (status == 429)


def test_stream_sends_status_events_then_the_answer():
    response = client_for(ScriptedService()).post("/chat/stream", json=BODY)
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "answer"]
    assert events[1][1]["response"]["answer"] == "N26 charges 10.00 EUR."


def test_stream_reports_a_failure_as_an_error_event():
    response = client_for(ScriptedService(error=ChatRateLimitError("slow down"))).post("/chat/stream", json=BODY)
    events = parse_sse(response.text)
    assert [name for name, _ in events] == ["status", "error"]
    assert events[-1][1] == {"type": "error", "code": "rate_limited", "message": "slow down"}


def test_stream_validates_the_request_too():
    assert client_for(ScriptedService()).post("/chat/stream", json={"session_id": "s1", "question": ""}).status_code == 422


def test_cors_allows_only_the_configured_origin():
    client = client_for(ScriptedService())
    headers = {"Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    allowed = client.options("/chat", headers={**headers, "Origin": ORIGIN})
    denied = client.options("/chat", headers={**headers, "Origin": "https://evil.example"})
    assert allowed.headers.get("access-control-allow-origin") == ORIGIN
    assert "access-control-allow-origin" not in denied.headers


def test_no_cors_headers_when_no_origins_are_configured():
    app = create_app(service=ScriptedService(), settings=Settings(_env_file=None, groq_api_key="x"))
    response = TestClient(app).get("/health", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def tight_limiter(**overrides):
    settings = dict(
        per_client_per_minute=50,
        global_per_minute=500,
        max_concurrent_per_client=5,
        max_concurrent_total=50,
        daily_budget=1000,
        lease_ttl_s=90,
    )
    settings.update(overrides)
    return RateLimiter(**settings)


def limited_client(service=None, limiter=None, **setting_overrides):
    settings = Settings(_env_file=None, groq_api_key="x", **setting_overrides)
    return TestClient(create_app(service=service or ScriptedService(), settings=settings, limiter=limiter))


@pytest.mark.parametrize("path", ["/chat", "/chat/stream"])
def test_a_visitor_over_the_rate_limit_gets_429_with_retry_after(path):
    client = limited_client(limiter=tight_limiter(per_client_per_minute=2))
    assert client.post(path, json=BODY).status_code == 200
    assert client.post(path, json=BODY).status_code == 200
    refused = client.post(path, json=BODY)
    assert refused.status_code == 429
    assert refused.json()["code"] == "too_many_requests"
    assert 1 <= int(refused.headers["retry-after"]) <= 60


def test_the_daily_budget_answers_429_with_its_own_code():
    client = limited_client(limiter=tight_limiter(daily_budget=1))
    assert client.post("/chat", json=BODY).status_code == 200
    refused = client.post("/chat", json=BODY)
    assert refused.status_code == 429 and refused.json()["code"] == "budget_exhausted"
    assert int(refused.headers["retry-after"]) > 0


def test_a_busy_server_answers_503():
    client = limited_client(limiter=tight_limiter(global_per_minute=1))
    client.post("/chat", json=BODY)
    refused = client.post("/chat", json=BODY)
    assert refused.status_code == 503 and refused.json()["code"] == "busy" and "retry-after" in refused.headers


def test_the_page_and_health_check_are_never_rate_limited():
    client = limited_client(limiter=tight_limiter(per_client_per_minute=1, daily_budget=1))
    client.post("/chat", json=BODY)
    assert client.post("/chat", json=BODY).status_code == 429
    assert client.get("/health").status_code == 200 and client.get("/").status_code == 200


def test_a_refused_request_never_reaches_the_service():
    class Counting(ScriptedService):
        calls = 0

        async def ask(self, request):
            Counting.calls += 1
            return await super().ask(request)

    client = limited_client(service=Counting(), limiter=tight_limiter(daily_budget=1))
    client.post("/chat", json=BODY)
    client.post("/chat", json=BODY)
    client.post("/chat", json=BODY)
    assert Counting.calls == 1


def test_finished_and_failed_requests_give_their_concurrency_slot_back():
    """With one slot per visitor, a leaked slot would make every later request fail."""
    limiter = tight_limiter(max_concurrent_per_client=1)
    ok = limited_client(limiter=limiter)
    for path in ("/chat", "/chat/stream", "/chat", "/chat/stream"):
        assert ok.post(path, json=BODY).status_code == 200, path

    failing = limited_client(service=ScriptedService(error=ChatServiceError("boom")), limiter=limiter)
    assert failing.post("/chat", json=BODY).status_code == 503  # the service failed...
    assert failing.post("/chat", json=BODY).status_code == 503  # ...and the slot was still returned (not 429)
    assert failing.post("/chat/stream", json=BODY).status_code == 200  # a stream reports failure as an event


def test_a_stream_holds_its_slot_while_running():
    limiter = tight_limiter(max_concurrent_per_client=1)
    client = limited_client(limiter=limiter)
    held = limiter.acquire("testclient")  # a stream that is still running for this visitor
    assert client.post("/chat", json=BODY).status_code == 429
    held.release()
    assert client.post("/chat", json=BODY).status_code == 200


def test_forwarded_addresses_are_only_trusted_when_configured():
    def send(client, forwarded):
        return client.post("/chat", json=BODY, headers={"X-Forwarded-For": forwarded}).status_code

    behind_proxy = limited_client(limiter=tight_limiter(per_client_per_minute=1), trusted_proxy_hops=1)
    assert send(behind_proxy, "9.9.9.1") == 200
    assert send(behind_proxy, "9.9.9.2") == 200  # a different visitor
    assert send(behind_proxy, "9.9.9.1") == 429  # the first one again

    not_behind_proxy = limited_client(limiter=tight_limiter(per_client_per_minute=1))
    assert send(not_behind_proxy, "9.9.9.1") == 200
    assert send(not_behind_proxy, "9.9.9.2") == 429  # forged header ignored: same socket address, same visitor


def test_a_server_without_an_llm_key_refuses_to_start(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    keyless = Settings(_env_file=None)
    with pytest.raises(MissingSecretError, match="GROQ_API_KEY"):
        create_app(settings=keyless)  # no service was supplied, so it would have to load the real one
    create_app(service=ScriptedService(), settings=keyless)  # a supplied service (tests) needs no key


def test_default_limits_come_from_settings():
    limiter = create_app(service=ScriptedService(), settings=Settings(_env_file=None, groq_api_key="x", rate_limit_per_minute=2, daily_request_budget=5)).state.limiter
    assert limiter._per_client == 2 and limiter._budget == 5


def test_sse_frame_format():
    assert format_sse(ChatEvent.status("hi")) == 'event: status\ndata: {"type":"status","message":"hi"}\n\n'


def test_the_api_works_end_to_end_with_the_real_service_and_keeps_history_per_session():
    service, condense, _ = make_service(FakeRetriever(DOCS), [GOOD])
    client = client_for(service)

    first = client.post("/chat", json=BODY).json()
    assert first["grounded"] and first["sources"][0]["page"] == 6
    assert condense.inputs == []  # first message of the session: nothing to rewrite

    second = parse_sse(client.post("/chat/stream", json={**BODY, "question": "and Revolut?"}).text)
    assert [name for name, _ in second] == ["status", "status", "status", "answer"]
    assert second[0][1]["message"] == "Reading the conversation so far..."
    assert len(condense.inputs) == 1  # the follow-up was rewritten using the first turn

    other = client.post("/chat", json={**BODY, "session_id": "someone-else"})
    assert other.status_code == 200 and len(condense.inputs) == 1  # a different session sees no history
