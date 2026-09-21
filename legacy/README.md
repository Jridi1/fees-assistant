# Earlier prototypes

The first versions of the assistant, kept as history. **They are not maintained and are not part of the deployed app.**

| File | What it was |
|---|---|
| `app.py` | Gradio app for Revolut only (the original Hugging Face Space) |
| `appV1.py`, `appV2.py` | Iterations toward multi-bank, with LangChain's conversational chain and one global memory |
| `dcapp.py` | Discord bot around `appV1.py` |

Why they were replaced by the `feesbot/` package: one conversation memory shared by every user, blocking calls inside an async Discord handler, an index rebuilt on every start, only N26 loaded, and sources listed even when the assistant refused. The reasoning is in [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

To run one you need extra packages (`gradio`, `discord.py`, `python-dotenv`, `langchain-community`, `pypdf`) and must start it from the repository root so it finds `N26/` and `internal_policy.pdf`.
