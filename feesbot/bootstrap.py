"""Wire settings, index, LLM and chat service together."""

from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq

from feesbot.chat import ChatService, build_answer_chain, build_condense_chain
from feesbot.ingestion import get_vectorstore
from feesbot.retrieval import build_retriever
from feesbot.sessions import SessionStore
from feesbot.settings import Settings, get_settings


def create_retriever(settings: Settings | None = None) -> BaseRetriever:
    """Retriever over the persisted index, with no LLM involved (for measuring recall)."""
    settings = settings or get_settings()
    return build_retriever(get_vectorstore(settings), None, settings)


def create_chat_service(settings: Settings | None = None, rebuild_index: bool = False) -> ChatService:
    """Build a ready-to-use `ChatService`. Call once at startup and share the instance."""
    settings = settings or get_settings()
    api_key = settings.require_groq_key()  # fail before building or loading anything
    store = get_vectorstore(settings, rebuild=rebuild_index)
    llm = ChatGroq(
        api_key=api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )
    return ChatService(
        retriever=build_retriever(store, llm, settings),
        condense_chain=build_condense_chain(llm),
        answer_chain=build_answer_chain(llm),
        sessions=SessionStore(settings.max_sessions, settings.max_history_turns),
        settings=settings,
    )
