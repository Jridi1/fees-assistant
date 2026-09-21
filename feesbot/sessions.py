"""In-memory, per-session conversation history with a bounded footprint."""

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One question/answer exchange."""

    question: str
    answer: str


class _Session:
    __slots__ = ("turns", "lock")

    def __init__(self, max_turns: int) -> None:
        self.turns: deque[Turn] = deque(maxlen=max_turns)
        self.lock = asyncio.Lock()


class SessionStore:
    """Conversation history keyed by session id.

    Each session keeps only its last `max_turns` exchanges, and at most
    `max_sessions` sessions are held: the least recently used one is evicted
    first (never one that is mid-request). This keeps memory bounded on a
    public endpoint. State is per process; use Redis or similar to share it
    across workers.
    """

    def __init__(self, max_sessions: int, max_turns: int) -> None:
        self._max_sessions = max_sessions
        self._max_turns = max_turns
        self._sessions: OrderedDict[str, _Session] = OrderedDict()

    def _get(self, session_id: str) -> _Session:
        session = self._sessions.get(session_id)
        if session is None:
            session = _Session(self._max_turns)
            self._sessions[session_id] = session
            self._evict()
        else:
            self._sessions.move_to_end(session_id)
        return session

    def _evict(self) -> None:
        while len(self._sessions) > self._max_sessions:
            # oldest first; the newest entry (last) is the one just created
            victim = next(
                (sid for sid, s in list(self._sessions.items())[:-1] if not s.lock.locked()),
                None,
            )
            if victim is None:  # everything older is busy: allow a temporary overshoot
                return
            del self._sessions[victim]

    def lock(self, session_id: str) -> asyncio.Lock:
        """Lock that serialises requests within one session, so history stays ordered."""
        return self._get(session_id).lock

    def history(self, session_id: str) -> list[Turn]:
        """Return a copy of the session's recent turns, oldest first."""
        return list(self._get(session_id).turns)

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Record a completed exchange."""
        self._get(session_id).turns.append(Turn(question, answer))

    def clear(self, session_id: str) -> None:
        """Forget a session."""
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)
