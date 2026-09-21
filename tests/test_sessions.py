import asyncio

from feesbot.sessions import SessionStore


def test_history_is_per_session_and_ordered():
    store = SessionStore(max_sessions=10, max_turns=5)
    store.add_turn("a", "q1", "a1")
    store.add_turn("a", "q2", "a2")
    store.add_turn("b", "other", "other")
    assert [t.question for t in store.history("a")] == ["q1", "q2"]
    assert [t.question for t in store.history("b")] == ["other"]


def test_history_returns_a_copy():
    store = SessionStore(10, 5)
    store.add_turn("a", "q", "a")
    store.history("a").clear()
    assert len(store.history("a")) == 1


def test_only_last_turns_are_kept():
    store = SessionStore(10, max_turns=2)
    for i in range(5):
        store.add_turn("a", f"q{i}", f"a{i}")
    assert [t.question for t in store.history("a")] == ["q3", "q4"]


def test_least_recently_used_session_is_evicted():
    store = SessionStore(max_sessions=2, max_turns=5)
    store.add_turn("a", "q", "a")
    store.add_turn("b", "q", "a")
    store.history("a")  # touch a, so b is now the oldest
    store.add_turn("c", "q", "a")
    assert len(store) == 2
    assert store.history("b") == []  # b was evicted (and is recreated empty here)
    assert store.history("c")


def test_busy_session_is_never_evicted():
    async def scenario():
        store = SessionStore(max_sessions=1, max_turns=5)
        async with store.lock("busy"):
            store.add_turn("busy", "q", "a")
            store.add_turn("new", "q", "a")  # over capacity, but "busy" is locked
            assert store.history("busy")
        return len(store)

    assert asyncio.run(scenario()) >= 1


def test_clear_forgets_the_session():
    store = SessionStore(10, 5)
    store.add_turn("a", "q", "a")
    store.clear("a")
    assert store.history("a") == []
