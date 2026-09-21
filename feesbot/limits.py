"""Abuse limits for a public endpoint: per-visitor and global rate limits, concurrency caps, a daily budget.

Every request costs LLM tokens and the provider's free tier is small, so an open endpoint can be
drained by one script. State is in memory and per process; with several workers it would move to a
shared store such as Redis. `acquire` has no awaits, so on a single asyncio event loop it is atomic.
"""

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

WINDOW_S = 60.0
DAY_S = 86400


class LimitExceeded(Exception):
    """A request was refused by a limit. `code` is machine-readable; `retry_after` is in seconds."""

    def __init__(self, code: str, detail: str, retry_after: float):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retry_after = max(1, math.ceil(retry_after))


@dataclass(eq=False)
class Lease:
    """A running request's claim on a concurrency slot. Release it when the request ends."""

    client: str
    started: float
    _release: Callable[["Lease"], None] = field(repr=False)

    def release(self) -> None:
        """Give the slot back. Safe to call more than once."""
        self._release(self)


def client_ip(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """Identify the visitor behind `trusted_hops` reverse proxies.

    Each proxy appends the address it received the request from, so the entry `trusted_hops` from
    the right is the visitor as seen by the outermost proxy we trust. Entries further left can be
    forged by the client and are ignored. With 0 hops the header is not trusted at all.
    """
    peer = peer or "unknown"
    if trusted_hops <= 0 or not forwarded_for:
        return peer
    parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
    if len(parts) < trusted_hops:
        return peer
    return parts[-trusted_hops]


class RateLimiter:
    """Checks, in order: per-visitor rate, per-visitor concurrency, global rate, global concurrency,
    daily budget. A refused request consumes nothing, so being refused never makes things worse."""

    def __init__(
        self,
        *,
        per_client_per_minute: int,
        global_per_minute: int,
        max_concurrent_per_client: int,
        max_concurrent_total: int,
        daily_budget: int,
        lease_ttl_s: float,
        clock: Callable[[], float] = time.time,
        max_clients: int = 10_000,
    ) -> None:
        self._per_client = per_client_per_minute
        self._global = global_per_minute
        self._conc_client = max_concurrent_per_client
        self._conc_total = max_concurrent_total
        self._budget = daily_budget
        self._lease_ttl_s = lease_ttl_s
        self._clock = clock
        self._max_clients = max_clients

        self._client_hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._global_hits: deque[float] = deque()
        self._leases: set[Lease] = set()
        self._day = -1
        self._used_today = 0

    # ---- public ----

    def acquire(self, client: str) -> Lease:
        """Admit a request or raise `LimitExceeded`. The caller must release the returned lease."""
        now = self._clock()
        self._drop_stale_leases(now)
        hits = self._client_hits.get(client)
        if hits is not None:
            self._prune(hits, now)
        self._prune(self._global_hits, now)
        self._roll_day(now)

        if hits is not None and len(hits) >= self._per_client:
            raise LimitExceeded("too_many_requests", "You are sending questions too quickly.", hits[0] + WINDOW_S - now)
        mine = sum(1 for lease in self._leases if lease.client == client)
        if mine >= self._conc_client:
            raise LimitExceeded("too_many_requests", "Please wait for your previous question to finish.", 3)
        if len(self._global_hits) >= self._global:
            raise LimitExceeded("busy", "The demo is busy right now.", self._global_hits[0] + WINDOW_S - now)
        if len(self._leases) >= self._conc_total:
            raise LimitExceeded("busy", "The demo is busy right now.", 3)
        if self._used_today >= self._budget:
            raise LimitExceeded("budget_exhausted", "The demo has reached its daily limit.", (self._day + 1) * DAY_S - now)

        if hits is None:
            hits = self._client_hits[client] = deque()
            self._evict_clients()
        else:
            self._client_hits.move_to_end(client)
        hits.append(now)
        self._global_hits.append(now)
        self._used_today += 1
        lease = Lease(client, now, self._release)
        self._leases.add(lease)
        return lease

    @property
    def used_today(self) -> int:
        """Requests admitted so far in the current UTC day."""
        return self._used_today

    # ---- internals ----

    def _release(self, lease: Lease) -> None:
        self._leases.discard(lease)

    def _drop_stale_leases(self, now: float) -> None:
        """A lease whose request vanished without releasing (a dropped connection) must not hold a slot forever."""
        for lease in [l for l in self._leases if now - l.started > self._lease_ttl_s]:
            self._leases.discard(lease)

    @staticmethod
    def _prune(hits: deque[float], now: float) -> None:
        while hits and hits[0] <= now - WINDOW_S:
            hits.popleft()

    def _roll_day(self, now: float) -> None:
        day = int(now // DAY_S)  # UTC days
        if day != self._day:
            self._day = day
            self._used_today = 0

    def _evict_clients(self) -> None:
        """Bound memory: forget the least recently seen visitors once there are too many."""
        while len(self._client_hits) > self._max_clients:
            self._client_hits.popitem(last=False)
