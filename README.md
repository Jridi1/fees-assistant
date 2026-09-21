---
title: Multi-bank Fees Assistant
emoji: 🏦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Bank fee answers from the banks' own PDFs, with citations
---

# Multi-bank fees assistant

A chatbot that answers questions about **N26 and Revolut fees only from the banks' own PDFs**, shows the page each answer came from, and says so when the documents do not cover the question.

**Live demo:** _(link added after the first deployment)_

> "How much does N26 charge for a replacement card?" answers **€10 for a standard card (N26 pricelist, page 5)**. A question the documents can't answer gets an honest "I don't have that information", with no invented number and no citation.

## Why it is built the way it is

Retrieval-augmented chatbots usually fail quietly: they answer confidently from the wrong passage. This project is built around catching that.

- **Grounded, cited, honest.** Every answer carries the bank, file and page it came from. If the documents don't contain the answer, the assistant refuses and cites nothing (enforced by the response model, not just the prompt).
- **Measured, not assumed.** A golden question set (facts read from the PDFs, with paraphrases and refusals) is run against both retrieval alone and the full pipeline. An early version paired a price with the wrong item; the evaluation is what exposed it and what proved the fix.
- **Hybrid retrieval.** Embedding search alone never surfaced the right price-list row. Keyword search plus embeddings, over small chunks widened with their neighbours, retrieves all 15 golden questions. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the diagrams and the decisions.
- **Built to be exposed.** Validated input (Pydantic), per-visitor conversation memory, streamed progress events, rate limits and a daily budget, safe rendering of model output, and a content-security policy.

## Run it locally

Requires Python 3.12 and a free [Groq API key](https://console.groq.com/keys).

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
cp .env.example .env                                        # then put your GROQ_API_KEY in it

python -m feesbot ingest      # build the search index once (a few minutes on first run)
python -m feesbot serve       # open http://127.0.0.1:8000/
```

Other commands: `python -m feesbot chat` (terminal chat), `python -m feesbot recall` (retrieval-only check, no API key), `python -m feesbot eval` (full live evaluation).

## Tests

```bash
python -m pytest tests        # unit, API and front-end tests (needs Node for the JavaScript ones)
python -m feesbot recall      # does retrieval surface every verified fact?
python -m feesbot eval        # does the assistant answer correctly, cite the right page, refuse when it should?
```

## Configuration

Set in `.env` or the environment. Everything except the key has a default.

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | **Required to serve.** Not needed to build the index |
| `CORS_ORIGINS` | `[]` | JSON list of sites allowed to embed the chat page in an iframe |
| `RATE_LIMIT_PER_MINUTE` | 8 | Questions per visitor per minute |
| `GLOBAL_RATE_LIMIT_PER_MINUTE` | 20 | Questions per minute across all visitors |
| `MAX_CONCURRENT_PER_CLIENT` / `MAX_CONCURRENT_TOTAL` | 2 / 6 | Questions running at once |
| `DAILY_REQUEST_BUDGET` | 300 | Questions per day in total (resets 00:00 UTC) |
| `TRUSTED_PROXY_HOPS` | 0 | Reverse proxies in front of the app; 0 ignores `X-Forwarded-For` |
| `LLM_MODEL`, `LLM_TEMPERATURE` | `openai/gpt-oss-120b`, 0.5 | The model used through Groq |
| `RETRIEVAL_K`, `USE_MULTI_QUERY` | 4, `false` | Retrieval breadth; optional LLM query expansion |
| `REQUEST_TIMEOUT_S` | 60 | Give up on a question after this long |

## Deploy

The [Dockerfile](Dockerfile) builds the search index into the image, so a sleeping host wakes in seconds. Step-by-step for Hugging Face Spaces: [docs/DEPLOY.md](docs/DEPLOY.md).

## Layout

```
feesbot/            the application
  api.py            FastAPI app: /chat, /chat/stream, /health, the chat page
  chat.py           ChatService: memory, retrieval, structured answer, citations
  hybrid.py         embeddings + BM25 search, neighbour windows
  chunking.py       PDF text -> chunks that know their pages
  ingestion.py      builds and reloads the Chroma index
  limits.py         rate limits, concurrency caps, daily budget
  schemas.py        validated data models   settings.py  typed configuration
  evaluation.py     golden-set evaluation
  static/           the chat page (HTML, CSS, JavaScript)
eval/golden.json    verified questions and expected answers
tests/              Python and JavaScript tests
docs/               architecture diagrams, deployment guide
legacy/             earlier prototypes (not maintained)
```

## Limitations

- Knows only the documents in the index (N26's document set and one Revolut policy). Answers can be wrong; the assistant says to check with the bank.
- The evaluation set is small (10 cases, 18 questions) and retrieval settings were tuned on the same facts, so the headline numbers are optimistic. Growing the set is the next quality step.
- Conversation memory and rate limits live in one process; running several workers would need a shared store such as Redis.
- `legacy/` holds the earlier prototypes (Gradio and Discord). They are kept as history and are not maintained.
- The bank PDFs in `N26/` and `internal_policy.pdf` belong to their publishers and are included only so the demo can answer from them.
