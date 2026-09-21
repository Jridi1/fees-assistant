import pytest

from feesbot.limits import DAY_S, LimitExceeded, RateLimiter, client_ip

START_OF_DAY = 20_000 * DAY_S  # a UTC midnight


class Clock:
    def __init__(self, t=START_OF_DAY + 100):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make(clock=None, **overrides):
    settings = dict(
        per_client_per_minute=3,
        global_per_minute=100,
        max_concurrent_per_client=10,
        max_concurrent_total=100,
        daily_budget=1000,
        lease_ttl_s=90,
    )
    settings.update(overrides)
    clock = clock or Clock()
    return RateLimiter(clock=clock, **settings), clock


def use(limiter, client="a"):
    limiter.acquire(client).release()


def refused(limiter, client="a"):
    with pytest.raises(LimitExceeded) as info:
        limiter.acquire(client)
    return info.value


# ---- per-visitor rate ----

def test_a_visitor_can_send_the_allowed_number_per_minute_then_is_refused():
    limiter, clock = make()
    for _ in range(3):
        use(limiter)
    error = refused(limiter)
    assert error.code == "too_many_requests" and error.retry_after == 60


def test_retry_after_counts_down_and_the_window_slides():
    limiter, clock = make()
    for _ in range(3):
        use(limiter)
    clock.advance(45)
    assert refused(limiter).retry_after == 15  # the oldest request leaves the window in 15 s
    clock.advance(16)
    use(limiter)  # allowed again


def test_visitors_are_limited_independently():
    limiter, _ = make()
    for _ in range(3):
        use(limiter, "a")
    refused(limiter, "a")
    use(limiter, "b")


# ---- concurrency ----

def test_a_visitor_cannot_run_more_requests_at_once_than_allowed():
    limiter, _ = make(max_concurrent_per_client=2, per_client_per_minute=50)
    first, second = limiter.acquire("a"), limiter.acquire("a")
    error = refused(limiter)
    assert error.code == "too_many_requests" and "previous question" in error.detail
    first.release()
    limiter.acquire("a")  # a slot is free again
    second.release()


def test_releasing_twice_frees_only_one_slot():
    limiter, _ = make(max_concurrent_per_client=2, per_client_per_minute=50)
    first, _second = limiter.acquire("a"), limiter.acquire("a")
    first.release()
    first.release()  # must not free the other request's slot
    limiter.acquire("a")
    refused(limiter)


def test_total_concurrency_is_capped_across_visitors():
    limiter, _ = make(max_concurrent_total=2, per_client_per_minute=50)
    limiter.acquire("a")
    limiter.acquire("b")
    error = refused(limiter, "c")
    assert error.code == "busy"


def test_a_lease_that_was_never_released_expires_instead_of_blocking_forever():
    limiter, clock = make(max_concurrent_per_client=1, per_client_per_minute=50, lease_ttl_s=10)
    limiter.acquire("a")  # e.g. the client vanished mid-stream and nothing released it
    refused(limiter)
    clock.advance(11)
    limiter.acquire("a")


# ---- global rate and daily budget ----

def test_the_global_rate_limit_counts_all_visitors_together():
    limiter, clock = make(global_per_minute=3, per_client_per_minute=50)
    for client in "abc":
        use(limiter, client)
    error = refused(limiter, "d")
    assert error.code == "busy" and error.retry_after == 60
    clock.advance(61)
    use(limiter, "d")


def test_the_daily_budget_is_spent_and_resets_at_utc_midnight():
    limiter, clock = make(daily_budget=2, per_client_per_minute=50)
    use(limiter, "a")
    use(limiter, "b")
    error = refused(limiter, "c")
    assert error.code == "budget_exhausted"
    assert error.retry_after == DAY_S - 100  # the clock started 100 s after midnight

    clock.advance(DAY_S)  # next day
    use(limiter, "c")
    assert limiter.used_today == 1


def test_a_refused_request_consumes_no_budget():
    limiter, _ = make(daily_budget=2, per_client_per_minute=1)
    use(limiter, "a")  # budget 1 of 2
    for _ in range(5):
        refused(limiter, "a")  # over the visitor limit
    use(limiter, "b")  # the rejections above did not eat the second unit
    assert limiter.used_today == 2
    assert refused(limiter, "c").code == "budget_exhausted"


# ---- memory bound ----

def test_the_table_of_visitors_stays_bounded():
    limiter, _ = make(max_clients=3, per_client_per_minute=50)
    for i in range(20):
        use(limiter, f"visitor-{i}")
    assert len(limiter._client_hits) <= 3


# ---- who is the visitor? ----

@pytest.mark.parametrize(
    "peer, forwarded, hops, expected",
    [
        ("10.0.0.1", "1.2.3.4", 0, "10.0.0.1"),  # header is ignored unless we trust a proxy
        ("10.0.0.1", "1.2.3.4", 1, "1.2.3.4"),
        ("10.0.0.1", "6.6.6.6, 1.2.3.4", 1, "1.2.3.4"),  # the client forged the left entry; the proxy appended the truth
        ("10.0.0.1", "1.2.3.4, 172.16.0.5", 2, "1.2.3.4"),  # two proxies
        ("10.0.0.1", "1.2.3.4", 2, "10.0.0.1"),  # fewer entries than proxies: do not trust it
        ("10.0.0.1", "", 1, "10.0.0.1"),
        ("10.0.0.1", None, 1, "10.0.0.1"),
        (None, None, 0, "unknown"),
    ],
)
def test_client_ip(peer, forwarded, hops, expected):
    assert client_ip(peer, forwarded, hops) == expected
