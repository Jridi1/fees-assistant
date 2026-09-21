"""Command line: `python -m feesbot ingest [--rebuild]` and `python -m feesbot chat`."""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from feesbot.chat import ChatServiceError
from feesbot.schemas import ChatRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="feesbot")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="build the vector index")
    ingest.add_argument("--rebuild", action="store_true", help="discard the existing index first")
    sub.add_parser("chat", help="talk to the assistant in the terminal")
    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    recall = sub.add_parser("recall", help="check retrieval finds each golden fact (no LLM, no API key)")
    recall.add_argument("--golden", type=Path, default=Path("eval/golden.json"))
    evaluate = sub.add_parser("eval", help="run the golden question set (calls the LLM)")
    evaluate.add_argument("--golden", type=Path, default=Path("eval/golden.json"))
    evaluate.add_argument("--case", action="append", help="run only this case id (repeatable)")
    evaluate.add_argument("--delay", type=float, default=8.0, help="seconds to wait between questions (rate limits)")
    return parser.parse_args()


async def _run_eval(golden: Path, only: list[str] | None, delay: float) -> int:
    from feesbot.bootstrap import create_chat_service
    from feesbot.evaluation import format_report, load_golden, run_eval

    cases = load_golden(golden)
    if only:
        cases = [c for c in cases if c.id in only]
        if not cases:
            print(f"No case matches {only}")
            return 2
    results = await run_eval(create_chat_service(), cases, delay_s=delay)
    print(format_report(results))
    return 0 if all(r.passed for r in results) else 1


async def _chat_loop() -> None:
    from feesbot.bootstrap import create_chat_service

    service = create_chat_service()
    print("Ask about bank fees (Ctrl+C or empty line to quit).")
    while True:
        try:
            line = (await asyncio.to_thread(input, "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        try:
            response = await service.ask(ChatRequest(session_id="cli", question=line))
        except ChatServiceError as exc:
            print(f"[error] {exc}")
            continue
        print(f"\n{response.answer}")
        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(f"  - {source.label}")
        elif not response.grounded:
            print("\n(not found in the documents)")


def main() -> None:
    """Parse the command line and run the chosen subcommand (ingest, serve, chat, recall or eval)."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "sentence_transformers", "langchain_classic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # hides the HF_TOKEN notice; read when HF is first imported
    args = _parse_args()
    if args.command == "ingest":
        from feesbot.ingestion import get_vectorstore
        from feesbot.settings import get_settings

        get_vectorstore(get_settings(), rebuild=args.rebuild)
        print("Index ready.")
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("feesbot.api:create_app", factory=True, host=args.host, port=args.port)
    elif args.command == "recall":
        from feesbot.bootstrap import create_retriever
        from feesbot.evaluation import evaluate_retrieval, format_retrieval_report, load_golden

        results = evaluate_retrieval(create_retriever(), load_golden(args.golden))
        print(format_retrieval_report(results))
        raise SystemExit(0 if all(r.passed for r in results) else 1)
    elif args.command == "eval":
        raise SystemExit(asyncio.run(_run_eval(args.golden, args.case, args.delay)))
    else:
        asyncio.run(_chat_loop())


if __name__ == "__main__":
    main()
