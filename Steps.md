Perfect choice — this is the smartest path. You'll learn faster, spend less, and still get full AWS experience by Week 5.

Let me give you **Week 1 in full beginner detail** — every concept explained, every command shown, every "why" answered so you can speak to an interviewer about it.

---

# 🎯 Week 1 — Build the Foundation (Local + Bedrock)

## What You'll Actually Build This Week

By end of Week 1, you'll have a **command-line tool** that:
1. Takes a PDF file as input
2. Extracts the text
3. Splits it into smaller pieces ("chunks")
4. Stores those chunks in a local Postgres database

**No web UI yet. No LLM yet. No embeddings yet.** Just a solid data pipeline. This is the boring-but-critical foundation.

## Why Start Here?

When an interviewer asks *"walk me through your RAG project"*, they want to hear:
> "It starts with ingestion. Users upload PDFs, I extract text using PyMuPDF, chunk it into ~400 token pieces with overlap, embed each chunk using Titan, and store both the text and the vector in Postgres with pgvector..."

You can't say that confidently unless you've **built it yourself, hit problems, and solved them.** That's Week 1.

---

## 🧠 Concepts You Need to Understand FIRST (Read this section twice)

Before any code, internalize these. An interviewer will ask about each.

### 1. What is RAG?

**Problem:** LLMs like Claude have a "knowledge cutoff" and don't know your private documents. If you ask "what does section 4.2 of my company's HR policy say?", Claude has no idea.

**Solution:** RAG = **R**etrieval **A**ugmented **G**eneration.

Three steps:
- **Retrieval** → Find the most relevant chunks of your documents
- **Augmentation** → Stuff those chunks into the prompt
- **Generation** → Let the LLM answer using that context

Think of it as **"giving the LLM an open-book exam"** instead of asking it from memory.

### 2. Why do we chunk documents?

You can't paste a 300-page PDF into a prompt — too expensive, too slow, and LLMs lose accuracy with very long contexts (this is the "Lost in the Middle" problem).

So we **split documents into smaller pieces (chunks)**, find only the relevant chunks, and send those.

**Tradeoff:**
- Small chunks (100 tokens) → precise retrieval but lose context
- Large chunks (1000 tokens) → more context but noisier

Sweet spot for most cases: **300-500 tokens with 50-100 tokens overlap**.

**Overlap** matters because important information might span chunk boundaries.

### 3. What is an embedding?

An embedding is a **list of numbers (a vector) that represents the meaning of text**.

Example: "dog" might become `[0.2, -0.5, 0.8, ..., 0.1]` (1024 numbers).

Two pieces of text with similar meaning have **similar vectors** (close in mathematical distance). This lets us search by meaning, not just keywords.

We'll add embeddings in Week 2. For Week 1, we just store the raw text chunks.

### 4. What is pgvector?

Postgres is a normal SQL database. **pgvector is an extension** that adds a new column type: `vector`. It lets you store embeddings and do similarity search inside Postgres — no separate vector database needed.

**Why pgvector vs FAISS/Chroma/Qdrant?**
- You already know FAISS/Chroma → boring, won't impress
- Companies prefer Postgres because they already use it for everything else
- pgvector handles SQL filtering + vector search in one query (huge in production)
- Interview-worthy talking point

### 5. Why Docker for Postgres?

Docker lets you run software in **isolated containers** without installing it system-wide. We'll run Postgres in a container so:
- It doesn't pollute your laptop
- You can delete it cleanly
- The exact same setup runs on AWS later (RDS uses the same Postgres)

**Mental model:** Docker container = a tiny virtual computer running just Postgres.

---

## 📋 Pre-Week Setup (Day 0 — do this before Monday)

### Step 1: AWS Account + Bedrock Access (THE most important — do FIRST)

Bedrock model access approval takes **12-24 hours**, so request it immediately.

1. Create AWS account at aws.amazon.com (needs credit card, but Bedrock charges pennies for our usage)
2. Enable MFA on root user (Settings → Security credentials)
3. Create IAM user named `rag-dev`:
   - IAM → Users → Create user
   - Attach policies: `AmazonBedrockFullAccess`, `AmazonS3FullAccess` (for later)
   - Create access key → save the Access Key ID + Secret Access Key
4. **Set budget alerts** (Billing → Budgets):
   - Alert at $10, $25, $50
   - Send email to yourself
5. **Request Bedrock model access** (Console → Bedrock → Model access → Manage model access):
   - Region: **us-east-1**
   - Request: Claude 3.5 Haiku, Claude 3.5 Sonnet, Titan Text Embeddings V2
   - Submit and wait ~24 hours for approval email

### Step 2: Install Local Tools

```bash
# Python with uv (modern, faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Docker Desktop — download from docker.com (free for personal use)
# After install, run: docker --version

# AWS CLI v2 — download from aws.amazon.com/cli
aws --version  # should be 2.x

# Configure AWS credentials
aws configure --profile rag-dev
# Paste your Access Key ID, Secret, region=us-east-1, output=json
```

### Step 3: Accounts

- **LangSmith** (smith.langchain.com) → sign up → get API key (we'll use it from Week 2, but get it now)
- **GitHub** → create private repo `rag-assistant`

### Step 4: Download Sample Papers

Save these 5 arXiv PDFs to `~/rag-data/papers/`:

| File | arXiv ID | What it teaches |
|---|---|---|
| Attention Is All You Need | 1706.03762 | Foundational transformer |
| BERT | 1810.04805 | Bidirectional encoders |
| RAG | 2005.11401 | The original RAG paper (meta!) |
| Lost in the Middle | 2307.03172 | Why context length matters |
| RAGAS | 2309.15217 | How to evaluate RAG |

Just google "arxiv 1706.03762 pdf" and download each.

---

## 🗓️ Week 1 — Day by Day

Each day is **2-3 hours** of focused work. Don't rush.

---

### 📅 Day 1 (Monday) — Project Skeleton

**Goal:** Set up your project folder with clean structure. Understand what each piece is for.

**Time:** 2 hours

#### Create the project

```bash
mkdir rag-assistant && cd rag-assistant
git init
```

Create this folder structure:

```
rag-assistant/
├── backend/
│   ├── pyproject.toml          # Python dependencies
│   ├── .env                     # Secrets (DO NOT commit)
│   ├── .env.example             # Template for .env
│   ├── .gitignore
│   ├── src/
│   │   └── rag/
│   │       ├── __init__.py
│   │       ├── config.py        # Reads .env
│   │       ├── ingestion/       # Document processing
│   │       │   ├── __init__.py
│   │       │   ├── extract.py   # PDF → text
│   │       │   └── chunker.py   # text → chunks
│   │       └── db/              # Database access
│   │           ├── __init__.py
│   │           └── connection.py
│   ├── scripts/
│   │   └── ingest.py            # Main CLI entry point
│   └── tests/
│       └── test_chunker.py
├── docs/
│   ├── architecture.md
│   └── decisions/               # ADRs (architecture decision records)
├── LEARNINGS.md                 # Your daily journal
└── README.md
```

#### Initialize Python project

```bash
cd backend
uv init  # creates pyproject.toml
```

#### Install dependencies

```bash
uv add pymupdf langchain-text-splitters tiktoken psycopg[binary] python-dotenv pydantic-settings boto3
uv add --dev pytest ruff
```

**What each does (memorize this for interviews):**

| Library | Purpose |
|---|---|
| `pymupdf` | Extract text from PDFs (fastest pure-Python option) |
| `langchain-text-splitters` | Smart chunking with token awareness |
| `tiktoken` | OpenAI's tokenizer — used to count tokens accurately |
| `psycopg[binary]` | Postgres driver for Python (v3, modern) |
| `python-dotenv` | Load `.env` files |
| `pydantic-settings` | Type-safe config from env vars |
| `boto3` | AWS SDK for Python (we'll use for Bedrock in Week 2) |
| `pytest` | Testing framework |
| `ruff` | Fast Python linter/formatter |

#### Create `.gitignore`

```
.env
__pycache__/
*.pyc
.venv/
.pytest_cache/
.DS_Store
*.egg-info/
```

#### Create `.env.example` (commit this)

```
# Postgres (local Docker for Weeks 1-2)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragdb
DB_USER=postgres
DB_PASSWORD=devpass

# AWS (we'll use these from Week 2)
AWS_PROFILE=rag-dev
AWS_REGION=us-east-1

# LangSmith (Week 2)
LANGSMITH_API_KEY=
```

Copy it: `cp .env.example .env` (this real one stays gitignored).

#### Commit

```bash
git add .
git commit -m "Day 1: project skeleton"
```

#### Write today's LEARNINGS.md entry

```markdown
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
```

---

### 📅 Day 2 (Tuesday) — Run Postgres in Docker

**Goal:** Have a working local Postgres database with the pgvector extension.

**Time:** 2 hours

#### Concept: what we're doing

We need a database to store our chunks. Instead of installing Postgres on your machine, we'll run it inside a **Docker container** — a sandboxed mini-environment.

The image `pgvector/pgvector:pg16` is **Postgres 16 with pgvector pre-installed**.

#### Start the container

```bash
docker run --name rag-pg \
  -e POSTGRES_PASSWORD=devpass \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 \
  -v rag-pg-data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16
```

**Decode that command (interview-ready explanation):**

| Flag | Meaning |
|---|---|
| `--name rag-pg` | Container name (so we can refer to it later) |
| `-e POSTGRES_PASSWORD=devpass` | Set the postgres user's password |
| `-e POSTGRES_DB=ragdb` | Create a database called `ragdb` on startup |
| `-p 5432:5432` | Map container port 5432 to your laptop's 5432 |
| `-v rag-pg-data:/var/lib/postgresql/data` | Persistent storage volume (data survives container restart) |
| `-d` | Detached mode (runs in background) |
| `pgvector/pgvector:pg16` | The Docker image to use |

**Verify it's running:**

```bash
docker ps
# You should see rag-pg in the list

docker logs rag-pg | tail
# Should say "database system is ready to accept connections"
```

#### Connect with psql

If you don't have `psql` installed, use the container's:

```bash
docker exec -it rag-pg psql -U postgres -d ragdb
```

You should see:

```
ragdb=#
```

#### Enable pgvector and create your tables

Paste this into psql:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per uploaded PDF
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    total_pages INT,
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

-- Many chunks per document
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_num INT,
    content TEXT NOT NULL,
    embedding vector(1024),    -- Titan v2 dimension; NULL for now
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for joining/filtering by document
CREATE INDEX chunks_doc_idx ON chunks(document_id);
```

Verify:

```sql
\dt
-- Should show 'documents' and 'chunks' tables

\d chunks
-- Should show the columns including 'embedding | vector(1024)'
```

Exit: `\q`

#### Save the schema to a file (so it's reproducible)

Create `backend/src/rag/db/schema.sql` with the SQL above. **This file is documentation — anyone can recreate your schema from it.**

#### Test connection from Python

Create a quick test in `backend/scripts/test_db.py`:

```python
import psycopg

conn = psycopg.connect(
    host="localhost",
    port=5432,
    dbname="ragdb",
    user="postgres",
    password="devpass",
)

with conn.cursor() as cur:
    cur.execute("SELECT version()")
    print(cur.fetchone())

conn.close()
```

Run it:

```bash
cd backend
uv run python scripts/test_db.py
```

You should see the Postgres version printed.

#### LEARNINGS.md Day 2

```markdown
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
```

---

### 📅 Day 3 (Wednesday) — PDF Text Extraction

**Goal:** Read a PDF and get clean text out of it.

**Time:** 2 hours

#### Why this is harder than it sounds

Real PDFs are messy:
- Scientific papers have **multi-column layouts**
- Some pages have **equations** (which often come out as garbled characters)
- Some PDFs are **scanned images** (need OCR — we'll handle in Week 3 with AWS Textract)
- **Tables** become spaghetti text

For Week 1, we use **PyMuPDF** which is fast and handles native-text PDFs well.

#### Write `extract.py`

`backend/src/rag/ingestion/extract.py`:

```python
"""PDF text extraction using PyMuPDF."""
from pathlib import Path
from typing import TypedDict
import pymupdf


class PageData(TypedDict):
    page_num: int
    text: str


class DocumentData(TypedDict):
    filename: str
    total_pages: int
    pages: list[PageData]


def extract_text(pdf_path: Path) -> DocumentData:
    """
    Extract text from a PDF, preserving page boundaries.
    
    Why per-page? So later we can cite "this answer came from page 5"
    in the final RAG response.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    doc = pymupdf.open(pdf_path)
    pages: list[PageData] = []
    
    for i, page in enumerate(doc):
        text = page.get_text()
        # Skip empty pages (cover pages, blanks)
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    
    return {
        "filename": pdf_path.name,
        "total_pages": len(doc),
        "pages": pages,
    }
```

#### Try it interactively

```bash
cd backend
uv run python
```

```python
from pathlib import Path
from rag.ingestion.extract import extract_text

data = extract_text(Path("~/rag-data/papers/1706.03762.pdf").expanduser())
print(f"Filename: {data['filename']}")
print(f"Pages: {data['total_pages']}")
print(f"First 500 chars of page 1:\n{data['pages'][0]['text'][:500]}")
```

**Observe the output carefully.** You'll notice:
- Math equations look weird (lots of unicode garbage)
- The abstract is readable
- Page numbers and headers leak into the text

**This is normal.** Don't fix it now. In production you'd clean these with regex, but for Week 1 we accept the messiness.

#### LEARNINGS.md Day 3

```markdown
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
```

---

### 📅 Day 4 (Thursday) — Chunking

**Goal:** Split extracted text into well-sized chunks for retrieval.

**Time:** 2 hours

#### Concepts revisited

Why ~400 tokens with overlap?

- **400 tokens** ≈ 300 English words ≈ 2-3 paragraphs. Big enough to contain a complete idea, small enough to be precise.
- **50 token overlap** prevents losing context at boundaries. If a sentence is cut in half, the next chunk still has the second half.
- **Recursive splitting** tries to break on paragraph → sentence → word boundaries (not random characters).

#### Write `chunker.py`

`backend/src/rag/ingestion/chunker.py`:

```python
"""Text chunking strategies."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def token_chunk(text: str, tokens: int = 400, overlap: int = 50) -> list[str]:
    """
    Split text into chunks of approximately `tokens` tokens
    with `overlap` tokens of overlap between consecutive chunks.
    
    Uses tiktoken for accurate token counting (matches OpenAI/Anthropic
    tokenization closely enough for retrieval purposes).
    
    Separators are tried in order: prefers paragraph breaks, then
    sentence breaks, then word breaks, then character.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=tokens,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)
```

#### Try it interactively

```bash
uv run python
```

```python
from pathlib import Path
from rag.ingestion.extract import extract_text
from rag.ingestion.chunker import token_chunk

data = extract_text(Path("~/rag-data/papers/1706.03762.pdf").expanduser())

# Chunk just page 2 (usually has abstract or intro)
chunks = token_chunk(data['pages'][1]['text'])
print(f"Page 2 produced {len(chunks)} chunks")
print("\n--- Chunk 0 ---")
print(chunks[0])
print("\n--- Chunk 1 ---")
print(chunks[1])
```

**Notice:** Compare the end of chunk 0 with the start of chunk 1. You should see some overlap. That's the safety net.

#### Write a real test

`backend/tests/test_chunker.py`:

```python
from rag.ingestion.chunker import token_chunk


def test_chunks_are_non_empty():
    text = "This is a test. " * 200
    chunks = token_chunk(text, tokens=50, overlap=10)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_chunks_have_overlap():
    """Consecutive chunks should share some text (proves overlap works)."""
    text = "Sentence one. Sentence two. " * 100
    chunks = token_chunk(text, tokens=50, overlap=10)
    # The end of chunk[0] should appear in chunk[1] somewhere
    end_of_first = chunks[0][-30:]
    # At least some words should reappear
    overlap_words = set(end_of_first.split()) & set(chunks[1].split())
    assert len(overlap_words) > 0


def test_small_text_returns_single_chunk():
    chunks = token_chunk("Just a tiny sentence.", tokens=400)
    assert len(chunks) == 1
```

Run:

```bash
uv run pytest tests/ -v
```

All 3 should pass.

#### LEARNINGS.md Day 4

```markdown
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
```

---

### 📅 Day 5 (Friday) — Config and DB Connection

**Goal:** Build the plumbing that connects Python to Postgres cleanly.

**Time:** 2 hours

#### Why a config module?

Hardcoding `host="localhost"` everywhere is bad. Production uses different values. **Pydantic Settings** reads from `.env`, validates types, and gives you autocomplete.

#### Write `config.py`

`backend/src/rag/config.py`:

```python
"""Application configuration from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "ragdb"
    db_user: str = "postgres"
    db_password: str = "devpass"
    
    # AWS (used from Week 2)
    aws_profile: str = "rag-dev"
    aws_region: str = "us-east-1"
    
    # LangSmith (Week 2)
    langsmith_api_key: str = ""


settings = Settings()
```

#### Write `connection.py`

`backend/src/rag/db/connection.py`:

```python
"""Postgres connection helper."""
from contextlib import contextmanager
from collections.abc import Iterator
import psycopg
from rag.config import settings


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """
    Yield a Postgres connection, closing it after use.
    
    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    conn = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )
    try:
        yield conn
    finally:
        conn.close()
```

**Interview point:** This is the **context manager pattern**. Connection is guaranteed to close even if an exception is raised.

#### Test it

```bash
uv run python -c "from rag.db.connection import get_connection; \
with get_connection() as c: \
    print('Connected:', c.info.dbname)"
```

Should print `Connected: ragdb`.

#### LEARNINGS.md Day 5

```markdown
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
```

---

### 📅 Day 6 (Saturday) — The Main Ingestion Script

**Goal:** Wire it all together. One command takes a PDF and stores chunks in DB.

**Time:** 3 hours

#### Write `ingest.py`

`backend/scripts/ingest.py`:

```python
"""
Ingest a PDF: extract text, chunk it, store in Postgres.

Usage:
    uv run python scripts/ingest.py path/to/paper.pdf
    uv run python scripts/ingest.py --dir ~/rag-data/papers
"""
import argparse
import logging
from pathlib import Path

from rag.ingestion.extract import extract_text
from rag.ingestion.chunker import token_chunk
from rag.db.connection import get_connection


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def ingest_pdf(pdf_path: Path) -> int:
    """Ingest one PDF. Returns the document_id."""
    log.info(f"Extracting {pdf_path.name}")
    data = extract_text(pdf_path)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Upsert document: same filename = same doc, replace chunks
            cur.execute(
                """
                INSERT INTO documents (filename, total_pages)
                VALUES (%s, %s)
                ON CONFLICT (filename)
                DO UPDATE SET total_pages = EXCLUDED.total_pages
                RETURNING id
                """,
                (data["filename"], data["total_pages"]),
            )
            doc_id = cur.fetchone()[0]
            
            # Remove old chunks (idempotent re-ingestion)
            cur.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))
            
            # Chunk per page so we keep page numbers
            global_chunk_idx = 0
            for page in data["pages"]:
                page_chunks = token_chunk(page["text"])
                for chunk_text in page_chunks:
                    cur.execute(
                        """
                        INSERT INTO chunks
                          (document_id, chunk_index, page_num, content)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (doc_id, global_chunk_idx, page["page_num"], chunk_text),
                    )
                    global_chunk_idx += 1
        conn.commit()
    
    log.info(f"✅ Stored doc_id={doc_id} with {global_chunk_idx} chunks")
    return doc_id


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("pdf", type=Path, nargs="?", help="Single PDF path")
    group.add_argument("--dir", type=Path, help="Directory of PDFs")
    args = parser.parse_args()
    
    if args.pdf:
        ingest_pdf(args.pdf)
    elif args.dir:
        pdfs = sorted(args.dir.expanduser().glob("*.pdf"))
        log.info(f"Found {len(pdfs)} PDFs in {args.dir}")
        for pdf in pdfs:
            ingest_pdf(pdf)


if __name__ == "__main__":
    main()
```

#### Run it

Make sure Postgres container is running:

```bash
docker ps
# If rag-pg isn't there: docker start rag-pg
```

Ingest all 5 papers:

```bash
cd backend
uv run python scripts/ingest.py --dir ~/rag-data/papers
```

You should see output like:

```
2026-05-12 14:23:01 INFO Extracting 1706.03762.pdf
2026-05-12 14:23:02 INFO ✅ Stored doc_id=1 with 87 chunks
2026-05-12 14:23:02 INFO Extracting 1810.04805.pdf
...
```

#### Verify in Postgres

```bash
docker exec -it rag-pg psql -U postgres -d ragdb
```

```sql
SELECT d.filename, COUNT(c.id) AS chunks
FROM documents d JOIN chunks c ON c.document_id = d.id
GROUP BY d.filename
ORDER BY d.filename;
```

Expected output:

```
       filename       | chunks
----------------------+--------
 1706.03762.pdf       |     87
 1810.04805.pdf       |    104
 2005.11401.pdf       |     91
 2307.03172.pdf       |     78
 2309.15217.pdf       |     65
```

Inspect a chunk:

```sql
SELECT content FROM chunks WHERE document_id = 1 LIMIT 1;
```

You should see real text from "Attention Is All You Need."

#### Run idempotently (test the upsert)

Run `ingest.py` again with the same dir. Chunks should be **replaced, not duplicated**. Verify counts are the same.

This is a **production-quality detail** — being able to safely re-run ingestion is huge.

#### LEARNINGS.md Day 6

```markdown
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
```

---

### 📅 Day 7 (Sunday) — Polish + Document + Reflect

**Goal:** Make this presentable. Diagram, README, ADR, video script.

**Time:** 2-3 hours

#### Task 1 — Architecture diagram

Open Excalidraw (excalidraw.com — free, no signup). Draw this:

```
[PDF File] ──> [ingest.py CLI]
                    │
                    ├──> PyMuPDF (extract text per page)
                    │
                    ├──> LangChain TextSplitter (chunk to 400 tokens)
                    │
                    └──> Postgres (Docker container)
                             ├── documents table
                             └── chunks table (embedding column NULL for now)
```

Export as PNG. Save to `docs/architecture-week1.png`.

#### Task 2 — Write your first ADR (Architecture Decision Record)

Senior engineers write ADRs. **Show this in interviews and you'll stand out.**

`docs/decisions/001-pgvector-over-dedicated-vector-db.md`:

```markdown
# ADR 001: Use pgvector instead of a dedicated vector database

## Status
Accepted (2026-05-12)

## Context
We need to store text embeddings and perform similarity search for RAG
retrieval. Options considered:
- pgvector (Postgres extension)
- Pinecone (managed SaaS)
- Qdrant / Weaviate (self-hosted)
- FAISS (in-memory library)

## Decision
Use pgvector on Postgres.

## Rationale
1. **Single database**: We already need Postgres for documents, users,
   audit logs. One DB = simpler ops, single backup strategy.
2. **Hybrid queries**: We can filter by metadata (e.g. user_id, document
   type) AND vector-search in one SQL query. Pinecone requires two
   round-trips or workarounds.
3. **AWS-native path**: RDS Postgres supports pgvector. No third-party
   service to integrate or pay for.
4. **Cost**: At ~500 chunks per document × 1024 dims, storage is trivial.
   Pinecone's free tier limits would force a paid plan quickly.

## Consequences
- We're capped by Postgres scale (~10M vectors comfortably with HNSW).
  For 100M+ vectors we'd need Pinecone or sharded pgvector.
- HNSW indexing is slower to build than FAISS, but query latency is
  comparable for our scale.

## Revisit when
- We exceed 5M chunks total
- Query p95 latency exceeds 500ms
```

#### Task 3 — Update the README

`README.md`:

```markdown
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
```

#### Task 4 — Write your interview elevator pitch

In `LEARNINGS.md`, draft a 60-second project intro for interviews:

```markdown
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
```

**Practice this out loud 5 times.** You should be able to say it without notes.

#### Task 5 — Commit and tag

```bash
git add .
git commit -m "Week 1 complete: ingestion pipeline"
git tag v0.1-ingestion
git remote add origin https://github.com/YOU/rag-assistant.git
git push -u origin main
git push --tags
```

#### Task 6 — Stop Docker (save laptop battery)

```bash
docker stop rag-pg
# Restart Monday with: docker start rag-pg
```

---

## ✅ End-of-Week-1 Checklist

```
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
```

---

## 🎤 Interview Questions You Can Now Answer

Practice each of these out loud — these are real questions you'll get:

1. **"Walk me through your ingestion pipeline."**
   → PDF → PyMuPDF per-page → LangChain RecursiveCharacterTextSplitter (400 tokens, 50 overlap) → Postgres with pgvector.

2. **"Why pgvector over Pinecone?"**
   → Single DB, hybrid queries, AWS-native path, no extra vendor.

3. **"Why 400 tokens for chunks?"**
   → Sweet spot: enough context for a complete idea, small enough for precise retrieval. Smaller is more precise but loses context, larger is noisier and more expensive.

4. **"What if the same PDF is uploaded twice?"**
   → ON CONFLICT upsert on filename, DELETE old chunks before inserting new — idempotent re-ingestion.

5. **"How does PyMuPDF handle scanned PDFs?"**
   → It doesn't. We'd detect them (very low text density) and route to OCR. On AWS that's Textract.

6. **"What's wrong with your current setup if I deploy it to production?"**
   → Synchronous, single-machine, no retries, secrets in .env, no observability. All addressed in later weeks.

---

## 🆘 Common Issues and Fixes

| Problem | Cause | Fix |
|---|---|---|
| `Connection refused` on port 5432 | Docker not running | `docker start rag-pg` |
| `password authentication failed` | Wrong password in .env | Check .env matches Docker `-e POSTGRES_PASSWORD` |
| `extension "vector" is not available` | Wrong image | Use `pgvector/pgvector:pg16`, not `postgres:16` |
| `ImportError: psycopg2` | Wrong package | `uv add psycopg[binary]` (v3, not v2) |
| Garbled text in chunks | Equations in PDF | Normal, will improve with cleaner PDFs |

---

## 📬 When You're Done

Ping me with:
1. Screenshot of the `SELECT d.filename, COUNT...` query output
2. Your `LEARNINGS.md`
3. Any concept that's still fuzzy

I'll send Week 2 (Bedrock embeddings + retrieval + FastAPI) — that's where it gets really fun. 🚀

**Go build. Don't read ahead. Focus on understanding every piece this week.**