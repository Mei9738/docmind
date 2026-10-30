# DocMind

A RAG service that answers questions about the FastAPI documentation, with cited sources.

**Live demo:** [docmind-70yb.onrender.com/docs](https://docmind-70yb.onrender.com/docs). Try `POST /ask` in the browser. It runs on a free plan that sleeps when idle, so the first request can take about a minute.

Built from scratch, without a RAG framework: every step of the pipeline (loading, chunking, embedding, retrieval, prompting) is hand-written and tested.

## How it works

**Ingestion** (`python -m docmind.ingest`)

1. **Load**: read every Markdown file in `data/`
2. **Chunk**: split each file on its headings; long sections are cut into fixed-size pieces with overlap
3. **Embed**: turn each chunk into a vector with `all-MiniLM-L6-v2` (sentence-transformers)
4. **Store**: save chunks, vectors and metadata (source file, heading, position) in ChromaDB

**Answering** (`POST /ask`)

1. **Retrieve**: embed the question and fetch the closest chunks (cosine distance), discarding any above a relevance threshold
2. **Prompt**: build a grounded prompt with numbered chunks and the question
3. **Generate**: ask an LLM (via Groq) to answer from that context only
4. **Respond**: return the answer, its sources and the retrieval scores

If no chunk is relevant enough, the service says so instead of calling the LLM.

## Project structure

```
src/docmind/
  loader.py      # Markdown files -> Document objects
  chunking.py    # Documents -> Chunks (headings + overlap split)
  ingest.py      # load -> chunk -> embed -> store in ChromaDB
  retrieval.py   # question -> top-k relevant chunks with distances
  prompts.py     # grounded prompt template
  generate.py    # prompt -> LLM answer + sources
  api.py         # FastAPI service
  config.py      # shared settings (models, collection, paths)
tests/           # pytest unit tests
data/            # corpus: FastAPI tutorial docs
```

## Getting started

Requires Python 3.11+ and a free [Groq](https://console.groq.com) API key.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
```

Create a `.env` file in the project root (it is git-ignored):

```
GROQ_API_KEY=your_key_here
```

Build the index, then start the API:

```bash
python -m docmind.ingest
fastapi dev src/docmind/api.py
```

Open http://127.0.0.1:8000/docs to try `POST /ask` in the browser, or:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I make a query parameter optional?"}'
```

## Run with Docker

No Python setup needed: Docker installs everything, builds the search index and starts the API.

Copy the example environment file, then put your Groq key in `.env`:

```bash
copy .env.example .env              # Windows (macOS/Linux: cp .env.example .env)
```

Build and start the service:

```bash
docker compose up
```

The first build takes a few minutes, because it builds the search index. Then open http://localhost:8000/docs.

## Tests

```bash
pytest
```

## Data

The corpus in `data/` is a subset of the [FastAPI documentation](https://github.com/fastapi/fastapi), © Sebastián Ramírez, used under the MIT License (see `data/FASTAPI_LICENSE.txt`). Code examples in those docs are referenced by placeholders and are not included.
