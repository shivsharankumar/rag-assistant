□ AWS account hardened (MFA, IAM user, $25 budget alert)
□ Bedrock model access approved (Haiku, Sonnet, Titan v2)
□ 5 arXiv papers downloaded
□ Python project structured with src/rag/ layout
□ Docker Postgres + pgvector running
□ Schema: documents + chunks tables created
□ extract.py: PDF → per-page text
□ chunker.py: text → 400-token chunks with overlap
□ config.py: Pydantic settings reading .env
□ connection.py: context-managed psycopg connections
□ ingest.py: idempotent CLI that wires it all together
□ 3+ pytest tests passing
□ Architecture diagram (Excalidraw)
□ ADR 001 written (pgvector decision)
□ README updated with Week 1 status
□ 7 LEARNINGS.md daily entries + 60-sec pitch
□ Git tagged v0.1-ingestion + pushed to GitHub

□ Bedrock embed_text() and claude_invoke() working
□ All 248 chunks have embeddings stored
□ HNSW index created on chunks.embedding
□ retrieve() returns relevant chunks with scores + metadata
□ synthesize_answer() produces grounded answers with citations
□ FastAPI /ask endpoint with Pydantic schemas
□ /health endpoint (for ALB later)
□ /docs OpenAPI UI working
□ LangSmith tracing on all 4 key functions
□ Traces visible in smith.langchain.com
□ eval/dataset.jsonl with 5 questions
□ Baseline retrieval accuracy measured
□ Architecture diagram updated
□ ADR 002 written
□ README updated
□ 90-second pitch practiced 5+ times
□ Git tagged v0.2-retrieval, pushed

□ chunks.content_tsv generated column + GIN index
□ hybrid_retrieve() with RRF fusion working
□ AgentState TypedDict defined
□ 7 graph nodes implemented (router, rewriter, retriever, grounding, synth, 2 apologies)
□ Compiled graph with conditional edges
□ /ask/stream endpoint with SSE
□ Eval shows hybrid > vector-only
□ Eval shows OOS questions correctly refused
□ Mermaid graph diagram saved
□ ADR 003 written
□ Architecture diagram updated
□ 90-second pitch revised + practiced
□ Git tagged v0.3-agent + pushed
□ Docker stopped to save laptop battery