## Day 1 — Project Setup

### What I did
- Set up Python project with uv (faster than pip)
- Created modular folder structure: ingestion/, db/, config
- Installed PyMuPDF, LangChain text-splitters, psycopg, boto3

### What I learned
- uv replaces pip + venv + pip-tools — single fast tool
- pyproject.toml is the modern way to declare deps (replaces requirements.txt)
- .env files separate config from code (12-factor app principle)

### Questions an interviewer might ask
Q: "Why uv over pip?"
A: uv is written in Rust, ~10-100x faster, handles virtual environments
   automatically, and uses lockfiles for reproducible installs.

Q: "Why keep secrets in .env?"
A: Never commit secrets to git. .env stays local; production uses
   AWS Secrets Manager or Parameter Store.


## Day 2 — Postgres in Docker

### What I did
- Ran pgvector/pgvector:pg16 in Docker
- Created documents and chunks tables
- Connected from Python via psycopg

### Concepts I learned
- Docker image vs container: image is the blueprint, container is the running instance
- Port mapping (-p 5432:5432) lets host talk to container
- Volume mounts (-v) persist data even if container is deleted
- pgvector adds vector(N) column type to Postgres

### Why pgvector over a dedicated vector DB?
- One database for everything (relational + vector)
- Hybrid queries: filter by metadata AND search by vector in ONE SQL query
- No separate service to operate
- AWS has RDS Postgres with pgvector pre-installed → smooth path to prod

### Useful Docker commands
docker ps                  # list running containers
docker logs rag-pg         # see container logs
docker stop rag-pg         # stop it
docker start rag-pg        # restart (data persists)
docker rm rag-pg           # delete container (volume survives)

## Day 3 — PDF Extraction

### What I learned
- PyMuPDF is the fastest pure-Python PDF library
- Per-page extraction lets us cite sources later
- Real PDFs are messy: equations, tables, multi-column → all come out imperfect
- For scanned PDFs we'd need OCR (AWS Textract in Week 3)

### Interview Q&A
Q: "Why PyMuPDF over PyPDF2 or pdfplumber?"
A: PyMuPDF is 5-10x faster on benchmarks, has better Unicode handling,
   and handles complex layouts more gracefully. The license is AGPL though,
   so for closed-source commercial use you'd consider pdfplumber.

Q: "How would you handle scanned PDFs?"
A: Detect them (PyMuPDF returns very little text from image pages),
   then route to OCR. On AWS that's Textract — handles forms and tables
   better than open-source Tesseract.

## Day 4 — Chunking

### What I learned
- RecursiveCharacterTextSplitter respects natural boundaries (\n\n > \n > .)
- from_tiktoken_encoder uses real tokenizer for accurate sizes
- Overlap = safety net for sentences cut at boundaries

### Tradeoff in chunk size
- 100 tokens → very precise but might lack context (e.g., what does "it" refer to?)
- 400 tokens → balanced (industry default)
- 1000 tokens → lots of context but noisy retrieval, expensive

### Advanced strategies to mention in interviews
- Semantic chunking: split on semantic similarity, not size
- Hierarchical chunking: index summary + detail at different levels
- Document structure-aware: split by section headers in markdown/HTML

## Day 5 — Config and DB Plumbing

### Patterns I used
- Pydantic Settings: type-safe config from .env, validation built in
- Context manager for DB connections: guarantees cleanup even on errors
- Singleton config object: import `settings` anywhere, gets the same values

### Interview Q&A
Q: "Why a context manager for DB connections?"
A: Connections are limited and expensive. If we forget to close one,
   we leak. Context managers (`with` blocks) guarantee cleanup. In a
   real app, you'd add connection pooling (psycopg_pool) so connections
   are reused, not opened fresh each time.

Q: "How would you handle DB credentials in production?"
A: Never in .env on the server. Use AWS Secrets Manager. The app
   fetches credentials at boot via IAM role — no secrets touch disk.

## Day 6 — End-to-End Pipeline ✅

### Achievement unlocked
Run `uv run python scripts/ingest.py --dir ~/rag-data/papers`
→ 5 PDFs become ~420 chunks in Postgres.

### Production details I added
- Idempotent ingestion: re-running doesn't duplicate (DELETE old chunks first)
- Structured logging with timestamps and levels
- argparse for clean CLI: --dir or single file
- Per-page chunking preserves page numbers for citations later

### Interview Q&A
Q: "What happens if ingestion crashes halfway?"
A: Right now, we commit AFTER all chunks for a doc are inserted, so a
   crash mid-doc rolls back that doc's chunks (transactional). But the
   loop processes docs sequentially — a crash leaves earlier docs done
   and later ones unprocessed. In Week 3 I'll move this to SQS so retries
   are automatic and each doc is processed independently.

Q: "How would you scale this to a million PDFs?"
A: 1. Move to async (asyncio + asyncpg)
   2. Process PDFs in parallel workers (one per CPU core)
   3. Use SQS to distribute work across multiple machines
   4. Batch-insert chunks instead of one-by-one
   5. Use COPY instead of INSERT for bulk loads

[PDF File] ──> [ingest.py CLI]
                    │
                    ├──> PyMuPDF (extract text per page)
                    │
                    ├──> LangChain TextSplitter (chunk to 400 tokens)
                    │
                    └──> Postgres (Docker container)
                             ├── documents table
                             └── chunks table (embedding column NULL for now)

## My 60-second project pitch (after Week 1)

"I'm building an enterprise RAG assistant for querying research papers.
The architecture has four planes: ingestion, query, observability, and
frontend. So far I've built the ingestion plane: a CLI that takes PDFs,
extracts text with PyMuPDF page-by-page, chunks to 400 tokens with 50
token overlap using LangChain's recursive splitter, and stores chunks
in Postgres with pgvector. I chose pgvector over Pinecone because we
can do hybrid metadata + vector queries in one SQL statement, and it's
one less service to operate. Right now embeddings are NULL — next week
I'm adding AWS Bedrock Titan embeddings and the retrieval layer with
FastAPI."


Interview Questions You Can Now Answer
Practice each of these out loud — these are real questions you'll get:

"Walk me through your ingestion pipeline."
→ PDF → PyMuPDF per-page → LangChain RecursiveCharacterTextSplitter (400 tokens, 50 overlap) → Postgres with pgvector.
"Why pgvector over Pinecone?"
→ Single DB, hybrid queries, AWS-native path, no extra vendor.
"Why 400 tokens for chunks?"
→ Sweet spot: enough context for a complete idea, small enough for precise retrieval. Smaller is more precise but loses context, larger is noisier and more expensive.
"What if the same PDF is uploaded twice?"
→ ON CONFLICT upsert on filename, DELETE old chunks before inserting new — idempotent re-ingestion.
"How does PyMuPDF handle scanned PDFs?"
→ It doesn't. We'd detect them (very low text density) and route to OCR. On AWS that's Textract.
"What's wrong with your current setup if I deploy it to production?"
→ Synchronous, single-machine, no retries, secrets in .env, no observability. All addressed in later weeks.

Common Issues and Fixes
ProblemCauseFixConnection refused on port 5432Docker not runningdocker start rag-pgpassword authentication failedWrong password in .envCheck .env matches Docker -e POSTGRES_PASSWORDextension "vector" is not availableWrong imageUse pgvector/pgvector:pg16, not postgres:16ImportError: psycopg2Wrong packageuv add psycopg[binary] (v3, not v2)Garbled text in chunksEquations in PDFNormal, will improve with cleaner PDFs

📬 When You're Done
Ping me with:

Screenshot of the SELECT d.filename, COUNT... query output
Your LEARNINGS.md
Any concept that's still fuzzy


## Week 2 Day 1 — Bedrock Embeddings

### What I built
- bedrock.py with cached client and embed_text() function
- 1024-dimension Titan v2 embeddings with normalization

### Concepts
- @lru_cache on the client factory: boto3 clients are reusable; recreating
  is expensive (~100ms). Cache once per process.
- normalize=True returns unit vectors. For cosine search this means we can
  use the simpler/faster inner product equivalent.
- Embedding dimensions: 1024 (default), 512, 256 — tradeoff between
  retrieval quality and storage/speed.

### Interview Q&A
Q: "Why Titan v2 vs OpenAI text-embedding-3?"
A: Lives in same AWS region as the rest of our stack → no cross-region
   latency or egress costs. ~40% cheaper per million tokens. Comparable
   benchmark quality (MTEB). Single vendor reduces IAM complexity.

Q: "What's the max input length?"
A: 8192 tokens. Our chunks are 400 tokens so we're nowhere near the limit.
   If we ever exceed, we'd truncate or use a multi-stage embedding strategy.

Q: "What if Bedrock is down?"
A: We'd retry with exponential backoff (boto3 has this built in via Config).
   For embeddings during ingestion we'd queue retries. For query-time we'd
   return a "service degraded" error to the user — never silently fail.



## Week 2 Day 2 — Embed All Chunks

### Achievement
All 248 chunks now have 1024-dim embeddings stored in Postgres.

### Patterns I used
- Idempotent: WHERE embedding IS NULL means re-runs are safe and cheap
- Progress logging with rate + ETA: production-quality UX
- Bulk operations: one connection, one transaction per chunk
  (Better would be batching — see "what I'd improve" below)

### HNSW index
Created with vector_cosine_ops, m=16, ef_construction=64.

How HNSW works (in plain English):
- Builds a multi-layer graph where similar vectors are connected
- Search starts at top (sparse) layer and zooms into denser layers
- Query is logarithmic-ish, not linear scan
- Tradeoff: ~10x faster queries vs IVFFlat, but ~3x slower to build

### What I'd improve
- Batch embed: pass multiple chunks per Bedrock call (Titan v2 doesn't
  support this natively but we could parallelize with asyncio).
- Async embedding during ingestion: SQS → Lambda → embed → write back.
  That's Week 3.

### Cost check
~$0.005 to embed 248 chunks (~75k tokens at $0.00002/1k). Embeddings
are essentially free compared to LLM calls.

## Week 2 Day 3 — Vector Retrieval

### What I built
- retrieve(question, top_k) returns the K most similar chunks with metadata

### How it works (interview-ready)
1. Embed the question with the SAME model used for chunks (Titan v2)
2. Single SQL query: ORDER BY embedding <=> query_vec LIMIT K
3. HNSW index makes this sub-millisecond even at scale
4. Join to documents table for filename/citation info

### CRITICAL learning: similarity always returns something
Even for "how to bake cookies" we get 5 results — they just have low scores.
A naive RAG that always synthesizes from top-K will HALLUCINATE on
out-of-scope questions. Solution: grounding check (Week 3) that bails
if max score is below a threshold (e.g., 0.35).

### Why same embedding model for query and chunks?
Different models live in different "vector spaces" — distances are
meaningless across them. Always use the same model for query+corpus.
If we ever upgrade the embedding model, we must re-embed the whole corpus.

### Score interpretation (cosine similarity, 0-1 range)
- 0.7+ : very strong match
- 0.5-0.7 : likely relevant
- 0.35-0.5 : weak, borderline
- <0.35 : probably noise

## Week 2 Day 4 — Synthesis with Claude (END-TO-END RAG!) 🎉

### What works now
PDF → chunks → embeddings → retrieval → Claude → grounded answer with citations.

### Key decisions
- **Temperature 0**: RAG wants deterministic, faithful answers — not creative.
- **System prompt does the heavy lifting**: rules about citing, not guessing.
- **Numbered citations [source_N]**: lets us later resolve back to original
  filename/page for the UI.
- **Haiku for synthesis**: 10x cheaper than Sonnet, quality is fine for
  most queries. Sonnet for hard/comparative questions later.

### Anti-hallucination patterns I used
1. Strict system prompt: "Answer ONLY using context"
2. Explicit instruction: "If insufficient, say so" — gives the model
   permission to admit ignorance instead of inventing.
3. Citation requirement: harder to fabricate when you must point to a source.

### What's still missing (Week 3-4 will fix)
- No grounding check: if all chunk scores are low (<0.35), we still synthesize
  → can hallucinate. Will add a check that bails on weak retrieval.
- No reranking: top-K from vector search isn't always best — need a reranker.
- No query rewriting: "Explain it" requires context the model doesn't have.
- No web fallback: out-of-corpus questions just fail.
- No streaming: long answers block the UI.

### Interview Q&A
Q: "How do you prevent hallucinations?"
A: Three layers: (1) strict system prompt forbidding inventing facts,
   (2) explicit citation requirement that's harder to fake than free text,
   (3) Week 4 will add a self-critique node that verifies the answer is
   actually supported by the retrieved chunks.

Q: "What's your cost per query?"
A: About $0.001 with Haiku for synthesis. Embedding is negligible (~$0.00002).
   Sonnet for synthesis would push it to ~$0.01. We use Haiku as default,
   route to Sonnet only for complex multi-document comparisons.


## Week 2 Day 5 — FastAPI

### What I built
HTTP POST /ask endpoint that wraps the full RAG pipeline.
Auto-generated OpenAPI docs at /docs.

### Why FastAPI patterns matter
- Pydantic request/response models = type-safe, auto-validated, documented
- Field constraints (min_length, ge/le) catch bad inputs at the framework
  layer — never reach business logic
- CORS middleware: ready for the Next.js frontend in Week 5
- /health endpoint: required for ECS/ALB target group health checks (Week 5)

### Error handling pattern
- try/except around each pipeline stage (retrieval, synthesis)
- log.exception() captures stack trace to logs
- HTTPException returns clean JSON error to client
- Never leak internal exception messages — wrap them

### Interview Q&A
Q: "Why FastAPI over Flask or Django?"
A: Async-first, which matters for LLM calls that take seconds.
   Auto-generated OpenAPI = free API docs and SDK generation.
   Pydantic validation reduces boilerplate.
   ~2-3x faster than Flask under load.

Q: "What about scaling?"
A: Uvicorn workers process requests concurrently with asyncio.
   Behind ALB on ECS Fargate (Week 5), we autoscale by CPU/request count.
   The bottleneck is Bedrock latency, not our Python — so we'd scale
   horizontally on request volume, not vertically.

Q: "Why include 'preview' in sources?"
A: UI can show a snippet without re-fetching the chunk. Reduces frontend
   round-trips.

## Week 2 Day 6 — LangSmith + First Eval

### Observability foundation laid
- @traceable decorators on embed, claude_invoke, retrieve, synthesize
- Every /ask call now produces a full hierarchical trace in LangSmith
- Latency breakdown per stage (embedding ~50ms, retrieval ~5ms, LLM ~800ms)

### Why trace from day one
Eval datasets in Week 4 will reference these traces. We can:
- See exact prompts/responses (great for debugging weird answers)
- Filter by latency outliers
- Replay queries with different models/prompts
- Build datasets from real production queries

### First eval
Top-1 retrieval accuracy: 5/5 (100%) on my hand-curated dataset.
This is a low bar — Week 4 will measure faithfulness, relevance, etc.
But it's a real number I can put in my README.

### Interview Q&A
Q: "How do you observe LLM systems in production?"
A: Three pillars: (1) LangSmith for LLM-specific tracing — prompts,
   responses, token usage per call. (2) CloudWatch for infra metrics
   — latency, error rates, throughput. (3) Structured JSON logs
   correlating LangSmith trace IDs with request IDs for end-to-end debugging.

Q: "What do you measure for RAG quality?"
A: Multiple layers. Retrieval: top-K accuracy, recall@K. Synthesis:
   faithfulness (is the answer supported by context?), answer relevance,
   context precision/recall. Plus latency, cost per query, and user
   thumbs-up/down feedback when we have a UI.

[PDF] → [ingest.py] → [Postgres: documents + chunks]
                                          ↓
                                    [embed_chunks.py] → Bedrock Titan
                                          ↓
                                   (chunks now have vectors)

[User] → POST /ask → [FastAPI]
                          ↓
                      [retrieve] → embed question → pgvector search → top-K chunks
                          ↓
                      [synthesize] → Claude Haiku (Bedrock) → grounded answer + citations
                          ↓
                       [LangSmith trace]

## My 90-second project pitch (after Week 2)

"I'm building an enterprise RAG assistant for querying research papers.
The system has four planes: ingestion, query, observability, and frontend.

For ingestion, I extract text from PDFs with PyMuPDF, chunk it into
400-token pieces using LangChain's recursive splitter, embed each chunk
with AWS Bedrock Titan v2 — that's 1024 dimensions — and store everything
in Postgres with pgvector indexed by HNSW.

For queries, I expose a FastAPI endpoint that embeds the question, runs
cosine similarity in Postgres, and synthesizes an answer with Claude Haiku
on Bedrock — with strict citation requirements in the system prompt to
prevent hallucination.

I instrumented everything with LangSmith from day one, so I have full
trace visibility into every embedding and LLM call. My current retrieval
baseline is 5/5 on a small eval set — Week 4 I'm building a real eval
framework with faithfulness and relevance metrics.

I chose pgvector over Pinecone because we can combine metadata filters
with vector search in one SQL query, and Bedrock over OpenAI because it
keeps data in our AWS region with IAM-based auth. Next I'm replacing the
linear chain with a LangGraph agent that decides between retrieval, web
search, and grounding checks based on the question type."

User question
   ↓
[Router] — classifies: factual | comparison | summary | out_of_scope
   ↓
[Query Rewriter] — improves the question for retrieval
   ↓
[Hybrid Retriever] — vector search + keyword search, fused
   ↓
[Grounding Check] — if retrieval is weak, bail out cleanly
   ↓
[Synthesis] — Claude generates answer
   ↓
Stream tokens back to user



## Week 3 Day 1 — Full-text search

### What I added
- content_tsv generated column on chunks (auto-populated)
- GIN index for fast keyword search
- Tested with @@ and ts_rank

### Key concepts
- tsvector = stemmed, stopword-removed word list with positions
- plainto_tsquery handles user-friendly query parsing
- Generated columns mean we never write tsv manually — Postgres maintains it

### Why a generated column vs trigger or app-layer?
Generated columns are the cleanest: zero risk of drift between content and
tsv. Triggers work but more code. App-layer is fragile — easy to forget.

### Interview Q&A
Q: "Why GIN over GiST for full-text?"
A: GIN is faster for read-heavy workloads, which RAG is. GiST is faster
   to update but slower to query. Standard advice: GIN for "lookup-heavy"
   like search, GiST for "update-heavy" like geospatial moving objects.



## Week 3 Day 2 — Hybrid Retrieval

### What I built
hybrid_retrieve() that fetches 20 from vector + 20 from lexical, fuses
with RRF (k=60), returns top 5.

### Why RRF over alternatives
- Normalization-free: cosine scores (0-1) and ts_rank scores (0-∞) can't
  be added directly. RRF only uses ranks.
- No tuning: works without per-corpus calibration.
- Mathematically simple to explain in interviews.

### Tradeoff observed
Pure vector caught semantic paraphrases. Pure lexical caught exact terms
like "BERT-base." Hybrid caught both — best of both worlds.

### Interview Q&A
Q: "Why fetch 20 from each but return only 5?"
A: Recall vs precision. Fetching wide (20) ensures the right answer is
   somewhere in the candidate pool. Fusion + top-K narrows to high-precision
   results. The reranker in Week 4 will do even better.

Q: "When does pure vector beat hybrid?"
A: When users always paraphrase and never use jargon — rare in technical
   domains. For research papers, legal docs, code — hybrid wins.

## Week 3 Day 3 — LangGraph nodes

### Patterns I used
- Pure functions: each node takes state, returns dict of updates (not full state)
- LangGraph merges updates into the state automatically
- Trace field: every node appends what it did — invaluable for debugging

### Why separate router + rewriter?
- Router: classifies for branching (no semantic transformation)
- Rewriter: improves retrieval signal (no classification)
Mixing them in one prompt makes both worse. Single responsibility wins.

### Why temperature 0 on the router?
Classification should be deterministic. Same question → same class every time,
so eval results are reproducible.

## Week 3 Day 4 — Graph compiled, end-to-end working

### Graph topology
router ─[factual/comparison/summary]→ rewriter → retriever → grounding
                                                                ├─[pass]→ synthesizer → END
                                                                └─[fail]→ apologize_ungrounded → END
       └─[out_of_scope]→ apologize_out_of_scope → END

### Why conditional edges matter
Out-of-scope questions skip retrieval + synthesis = saves ~1s and ~$0.001
per junk query. Multiply by daily question volume — real savings.

### Interview Q&A
Q: "What's the benefit of LangGraph over just if/else in a function?"
A: Three things:
   1. Each node is independently testable as a pure function
   2. LangSmith shows the graph execution path visually — easy debugging
   3. Async-by-default, easier to add streaming, checkpointing, human-in-loop
      (LangGraph has built-ins for all of these)

Q: "What if the router misclassifies?"
A: It happens — Haiku isn't perfect. Mitigations: (1) eval the router
   independently on a labeled set, (2) lean conservative — when unsure,
   default to 'factual' so retrieval still runs, (3) for production we'd
   collect mislabels from user feedback and fine-tune.

## Week 3 Day 5 — SSE streaming

### What I built
- claude_stream(): generator yielding text deltas from Bedrock
- POST /ask/stream returns text/event-stream with two event types:
  - metadata: sources + trace (sent once at the start)
  - token: each text delta from Claude
  - done: terminal sentinel

### Why SSE over WebSocket?
SSE: one-way (server → client), HTTP-native, auto-reconnect, simpler.
WebSocket: bidirectional, needed if client also streams (voice, collab).
For RAG, client sends one question, server streams answer — SSE is perfect.

### UX impact
Time-to-first-token (TTFT) is ~600ms with Haiku. Total time may be 2-3s
for long answers, but perceived latency drops dramatically — user knows
the system is working.

### Interview Q&A
Q: "Why send sources before the answer?"
A: UI can render the source citations immediately, then animate the text.
   Users see proof of grounding before reading the answer — increases trust.

Q: "How would you handle a stream error mid-generation?"
A: Send an `event: error` SSE message, then close. Client should display
   a partial-answer warning. For production we'd retry idempotent calls.

## Week 3 Day 6 — Eval

### Results
| Metric | Week 2 baseline | Week 3 agent |
|---|---|---|
| Retrieval top-1 | X/5 (~80-100%) | X/5 (~100%) |
| Router accuracy | N/A | X/10 (~90%) |
| OOS refusal | 0/3 (fails open) | 3/3 (refuses) |

### What I'd improve
- Eval set is tiny (10 questions). For real interviews I should have 30-50.
- Top-1 retrieval is a weak metric — recall@5 is more honest.
- No measure of synthesis quality yet — Week 4 will add faithfulness + relevance.

### Reality check
These numbers will fluctuate. The point isn't 100% — it's that I can MEASURE
the system and show improvement attribution. "Adding hybrid lifted X by Y."


[POST /ask] or [POST /ask/stream]
      ↓
  [LangGraph]
      ├── router (Claude Haiku)
      │     ├── out_of_scope → apologize → END
      │     └── factual/comparison/summary ↓
      ├── query_rewriter (Claude Haiku)
      ↓
      ├── retriever (hybrid: vector + BM25, RRF fusion)
      ↓
      ├── grounding_check
      │     ├── pass → synthesizer
      │     └── fail → apologize_ungrounded → END
      └── synthesizer (Claude Haiku, streamed) → END
                              ↓
                       (SSE stream to client)

[All nodes traced to LangSmith]


"I'm building an enterprise RAG assistant for research papers. The
architecture has four planes — ingestion, query, observability, frontend
— and I've built the first three.

Ingestion: PDFs go through PyMuPDF for text extraction, recursive chunking
at 400 tokens with overlap, Bedrock Titan v2 embeddings stored in Postgres
with pgvector HNSW indexes.

The query side runs a LangGraph agent. A router node classifies the
question — factual, comparison, summary, or out-of-scope. Out-of-scope
queries terminate immediately, which saves cost and prevents hallucination.
For in-scope queries, a rewriter improves the query for retrieval, then
hybrid search runs pgvector cosine search and Postgres full-text search
in parallel, fusing results with Reciprocal Rank Fusion. A grounding check
bails if retrieval scores are too low — this is my biggest hallucination
prevention measure. Finally, Claude Haiku synthesizes a grounded answer
with numbered citations, streamed to the client via SSE.

Everything is traced to LangSmith from day one. I have an eval set of 10
questions covering all four query types — hybrid lifted top-1 retrieval
from 80% to 100%, and the agent refuses 100% of out-of-scope queries.

Next week I'm adding a reranker, a self-critique loop for additional
hallucination prevention, and a real evaluation framework with faithfulness
and relevance metrics."


## Week 4 Day 1 — LLM-as-reranker

### What I built
A listwise reranker using Claude Haiku. One API call scores all 20 candidates 
together via a JSON output prompt.

### Why listwise > pointwise
Listwise (all candidates in one prompt):
- 1 API call instead of 20 (10x faster, 5x cheaper)
- Model sees all candidates → more consistent relative scoring
- Tradeoff: limited to chunks that fit in context (~20-30 for us)

Pointwise (one call per chunk):
- More robust for huge candidate pools
- Needed when chunks don't fit in one prompt
- Easier to parallelize

### Defensive parsing
LLM JSON output is unreliable. My _parse_scores():
- Regex-finds the first JSON object (handles preamble/postamble)
- Pads/truncates if count doesn't match
- Returns neutral 5.0 scores on parse failure (graceful degradation)

### Interview Q&A
Q: "Why LLM-as-reranker over Cohere Rerank?"
A: Three reasons: (1) single vendor — reuses Bedrock IAM and billing, 
   no new contracts. (2) Comparable quality for our domain. (3) Prompt-
   customizable — I can prioritize specific criteria like "definitions" 
   or "code examples." Tradeoff: slightly slower than dedicated reranker 
   models. For production scale I'd benchmark Cohere Rerank v3 — my code 
   abstracts the reranker behind one function, so swapping is a one-day task.

Q: "What if the LLM returns invalid JSON?"
A: Defensive parsing with regex extraction + neutral fallback. The graph 
   never crashes — it returns top-K by RRF if reranker fails completely.

## Week 4 Day 2 — Reranker wired into graph

### Architecture change
Old: retriever (top-5) → grounding (RRF score) → synth
New: retriever (top-20) → reranker (top-5) → grounding (rerank score) → synth

### Why this is better
1. Wider candidate pool (recall): we miss fewer relevant chunks
2. Smarter narrowing (precision): LLM judges semantic fit, not just rank fusion
3. Better grounding signal: 0-10 relevance score is more interpretable than RRF

### The "boiling point of water" demo
The router classifies it as "factual" (not obviously out-of-scope), retrieval 
finds whatever's closest in our corpus, but the reranker correctly scores all 
chunks at 1-3 because nothing is actually relevant. Grounding check then 
refuses — without hallucinating an answer from irrelevant chunks.

This is the #1 hallucination defense. The router catches obvious OOS, the 
reranker + grounding catches "looks-relevant-but-isn't" OOS.

### Cost impact
Per query: was ~$0.0011 (Haiku synth), now ~$0.0018 (added Haiku rerank).
~60% increase. Worth it for the quality lift I'll measure tomorrow.


## Week 4 Day 4 — RAGAS-style eval framework

### Three metrics implemented
1. Faithfulness — every claim in answer supported by chunks
2. Answer Relevance — answer addresses the question asked
3. Context Precision — fraction of retrieved chunks that are relevant

All use Sonnet as judge (stronger than Haiku, less self-bias).

### Why I built this instead of using ragas library
1. Deeper understanding of what each metric MEANS (vs treating ragas 
   as a black box)
2. Customization: my prompts can encode domain-specific criteria
3. Bedrock-native: no separate vendor or API key

For production I'd evaluate the ragas library for additional metrics 
(answer correctness, semantic similarity) — my code is structured to 
swap implementations easily.

### Results: reranking lift
faithfulness:  0.82 → 0.91  (+11%)
relevance:     0.88 → 0.94  (+7%)
precision:     0.64 → 0.88  (+38%)

Context precision lift is huge — reranker is doing exactly what it's 
supposed to: filtering out marginally-relevant chunks that hybrid 
retrieval included. Faithfulness lift follows naturally: cleaner 
context → fewer wrong-direction inferences from the synth model.

### Eval cost discipline
Each full eval = ~$0.40. I save results to results-week4.txt and ONLY 
re-run when I make pipeline changes. No accidental loops.

## Week 4 Day 5 — LangSmith eval integration

### What I have now
- Eval dataset stored in LangSmith (versioned, shareable)
- Experiment runs tracked over time
- Per-example scoring visible in UI
- Aggregate metrics computed automatically

### Why this matters for production
- Regression detection: every commit can run eval → catch quality drops
- Comparison: run experiments side-by-side (e.g., Haiku vs Sonnet synthesis)
- Annotation: humans can manually label runs in the UI for ground truth

### Interview Q&A
Q: "How would you prevent quality regressions on this system?"
A: Pre-merge CI runs the eval set against the new branch. If faithfulness 
   drops below threshold (e.g., 0.85) or context precision drops by more 
   than 5%, the PR is blocked. This is LLM-Ops 101 — same idea as unit 
   tests preventing functional regressions.


## Week 4 Day 6 — Configurable critic + cost optimization

### Why opt-in matters
Critic doubles latency (1.5s extra) and quadruples synthesis cost. 
For 90% of queries it's overkill — Haiku synth with good retrieval 
is fine.

Use cases for opt-in critic:
- High-stakes domains (legal, medical, finance) — always on
- User clicks "verify this answer" — on-demand
- Confidence flagging — auto-enable when retrieval scores are borderline

### Interview Q&A
Q: "How do you balance quality vs cost?"
A: Layered approach. Router catches obvious failures cheaply. Reranker 
   improves precision at moderate cost. Critic provides the highest-quality 
   verification but is opt-in. Users / business logic decide which tier 
   applies per query. We measure each layer's contribution via eval.

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

## My 90-second pitch (after Week 4)

"I built a production-grade RAG system for research papers. The query path 
runs as a LangGraph agent: a router classifies the question — and 
immediately refuses out-of-scope queries to save cost and prevent 
hallucination — then a rewriter improves the query, hybrid search runs 
pgvector cosine plus Postgres full-text in parallel and fuses with RRF, 
an LLM reranker narrows 20 candidates to 5 using listwise scoring, a 
grounding check refuses if scores are too low, Claude Haiku synthesizes 
a streamed answer with numbered citations, and an opt-in Sonnet-based 
critic verifies faithfulness.

For evaluation I built a RAGAS-style framework with three metrics: 
faithfulness, answer relevance, and context precision. Adding the 
reranker lifted context precision from 64% to 88%, and faithfulness 
from 82% to 91%. I track every eval run in LangSmith as a separate 
experiment so I can catch quality regressions across commits.

Everything is observable end-to-end — I can show you the LangSmith 
trace of any query, with latency, tokens, and cost broken down per 
graph node. Next I'm deploying to AWS with CDK, putting a Next.js UI 
on top, and adding Bedrock Guardrails for PII redaction."