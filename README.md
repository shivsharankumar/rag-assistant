# RAG Assistant

A production-grade Retrieval-Augmented Generation system for querying
research papers. Built from scratch as a portfolio project.

## Status
🚧 Week 1/6: Local ingestion pipeline complete.

## Architecture (Week 1)
![Week 1 architecture](docs/architecture-week1.png)

## What works today
- PDF text extraction (PyMuPDF)
- Token-aware chunking with overlap (400 tokens, 50 overlap)
- Postgres + pgvector storage with proper schema
- Idempotent ingestion CLI

## Roadmap
- Week 2: Bedrock Titan embeddings + semantic retrieval + FastAPI
- Week 3: LangGraph agent, S3, RDS migration
- Week 4: Eval framework, reranking, self-critique
- Week 5: Next.js UI, ECS deployment via CDK, Cognito auth
- Week 6: Polish, blog post, demo video

## Tech stack
Python · PyMuPDF · LangChain · Postgres + pgvector · Docker

## Run locally
```bash
docker run --name rag-pg -e POSTGRES_PASSWORD=devpass \
  -e POSTGRES_DB=ragdb -p 5432:5432 -d pgvector/pgvector:pg16

cd backend
cp .env.example .env
uv sync
uv run python scripts/ingest.py --dir ~/rag-data/papers
```

## Architecture Decisions
- [ADR 001: pgvector over dedicated vector DB](docs/decisions/001-pgvector-over-dedicated-vector-db.md)

## Status
🚧 Week 2/6: Embeddings, retrieval, and HTTP API complete.

## What works today
- ✅ PDF ingestion (PyMuPDF, idempotent CLI)
- ✅ Token-aware chunking (400 tokens, 50 overlap)
- ✅ Bedrock Titan v2 embeddings (1024 dim) → pgvector HNSW index
- ✅ Cosine similarity retrieval (top-K with metadata + scores)
- ✅ Claude Haiku synthesis with grounded answers + numbered citations
- ✅ FastAPI HTTP layer with auto-generated OpenAPI docs
- ✅ LangSmith tracing on every LLM call
- ✅ Retrieval accuracy baseline: 5/5 on hand-curated eval set

## Roadmap
- Week 3: LangGraph agentic flow, S3 ingestion, RDS migration, streaming
- Week 4: Reranking, self-critique, full eval framework
- Week 5: Next.js UI, ECS deployment via CDK, Cognito auth
- Week 6: Polish, blog post, demo video

## Architecture
![Week 2 architecture](docs/architecture-week2.png)

## Decisions
- [ADR 001: pgvector over dedicated vector DB](docs/decisions/001-pgvector-over-dedicated-vector-db.md)
- [ADR 002: Bedrock Titan for embeddings](docs/decisions/002-bedrock-titan-for-embeddings.md)

## Try it
```bash
docker start rag-pg
cd backend
uv run uvicorn rag.api.main:app --reload
# Open http://localhost:8000/docs
```


## Status
🚧 Week 3/6: Agentic workflow + hybrid search + streaming.

## What works today
- ✅ All of Week 1-2
- ✅ Hybrid retrieval (pgvector + Postgres tsvector + RRF fusion)
- ✅ LangGraph agent: router → rewriter → retriever → grounding → synthesizer
- ✅ Out-of-scope handling (refuses without hallucinating)
- ✅ Grounding check (bails if retrieval too weak)
- ✅ SSE streaming via POST /ask/stream
- ✅ Eval: retrieval top-1 lifted from 80% to 100%, OOS refusal 0% → 100%

## Decisions
- [ADR 001: pgvector](docs/decisions/001-pgvector-over-dedicated-vector-db.md)
- [ADR 002: Bedrock Titan](docs/decisions/002-bedrock-titan-for-embeddings.md)
- [ADR 003: Hybrid search + LangGraph](docs/decisions/003-hybrid-search-and-langgraph.md)

[/ask] → [LangGraph]
            ├── router [Haiku]
            │     ├── out_of_scope → apologize → END
            │     └── factual/comparison/summary ↓
            ├── query_rewriter [Haiku]
            ├── retriever [hybrid: vector + BM25, RRF, top 20]
            ├── reranker [Haiku, listwise, JSON output, top 5]
            ├── grounding_check [uses rerank score]
            │     ├── pass → synthesizer
            │     └── fail → apologize_ungrounded → END
            ├── synthesizer [Haiku, with streaming]
            └── critic (optional) [Sonnet judge]
                  → faithfulness score
                  → unsupported claims surfaced
                  → END

[Eval: faithfulness, relevance, context precision via LangSmith]

## Quality Metrics (Week 4 baseline)

Measured on a 5-question eval set across 5 papers (RAGAS-style, LLM-as-judge).

| Metric | Without reranker | With reranker |
|---|---|---|
| Faithfulness | 0.82 | **0.91** |
| Answer Relevance | 0.88 | **0.94** |
| Context Precision | 0.64 | **0.88** |

Out-of-scope refusal: 100% (3/3) — no hallucinated answers to off-topic queries.

Full eval output: [eval/results-week4.txt](backend/eval/results-week4.txt)
Tracked in LangSmith: [experiment dashboard screenshot](docs/langsmith-eval.png)