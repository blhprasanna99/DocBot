# DocBot

A conversational AI documentation agent: ask questions in natural language, get grounded answers with sources cited from internal company docs.

Built with Python, FastAPI, LangChain, ChromaDB, and the Claude API. Indexed against the public PostHog handbook (~260 markdown pages) as a real-world stand-in for an internal knowledge base.

---

## What it does

```
question → embed → ChromaDB similarity search → top-K chunks
                                                      ↓
                                         Claude (grounded prompt)
                                                      ↓
                                          {reply, [source files]}
```

Every answer is grounded only in retrieved context. If the handbook doesn't contain the answer, DocBot says so instead of hallucinating.

## Stack

| Concern | Tool |
|---|---|
| Web framework | FastAPI + Uvicorn |
| LLM | Claude (Sonnet 4.6) via Anthropic API |
| RAG glue | LangChain |
| Vector store | ChromaDB (persisted to disk) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (local, no API cost) |
| Frontend | Vanilla HTML / CSS / JS (no build step) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

Fetch the corpus (sparse checkout of the public PostHog handbook):

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/PostHog/posthog.com.git docs/posthog
cd docs/posthog && git sparse-checkout set contents/handbook && cd ../..
```

Build the vector store (one-time, ~5–15 min on CPU):

```bash
python -m app.ingest
```

Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/` in a browser.

## Eval

A small offline eval harness (`eval/`) measures retrieval quality against a curated set of 12 questions with known ground-truth source files.

| Embedding model | hit@8 | Avg retrieval latency |
|---|---|---|
| `all-MiniLM-L6-v2` (baseline) | 58.3% | ~370 ms |
| `all-mpnet-base-v2` (current) | 75.0% | ~310 ms |

Run it with:

```bash
python eval/run_eval.py
```

The harness prints per-question hit/miss, average latency, and (for misses) what was retrieved instead — useful for tuning chunk size, K, or the embedding model.

## API

`POST /chat`

```json
{ "message": "How is feedback given at PostHog?" }
```

Response:

```json
{
  "reply": "...",
  "sources": ["docs/posthog/contents/handbook/people/feedback.md", "..."]
}
```

`GET /health` returns `{"ok": true}`.

Auto-generated OpenAPI docs at `/docs`.

## Project layout

```
app/
  main.py        FastAPI app: /, /chat, /health
  ingest.py      One-shot: load markdown → chunk → embed → persist Chroma
docs/posthog/    Corpus (gitignored; fetch instructions above)
eval/
  questions.yaml Ground-truth question/source pairs
  run_eval.py    Offline retrieval eval (no API calls, free to run)
static/
  index.html     Single-page chat UI
.chroma/         Persisted vector store (gitignored; rebuild with ingest)
```

## Known limitations

- Retrieval misses on highly abstract queries ("how does the company make money?") because the embedding model rewards lexical/topical proximity.
- No conversation memory — each `/chat` call is independent. History lives only in the browser session.
- No re-ingest deduplication — running `python -m app.ingest` twice will duplicate vectors. Delete `.chroma/` between runs.

## License

MIT (see `LICENSE`).
