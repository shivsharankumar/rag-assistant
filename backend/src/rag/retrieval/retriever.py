"""Vector similarity retrieval from Postgres."""
import logging
from typing import TypedDict
from src.rag.aws.bedrock import embed_text
from src.rag.db.connection import get_connection
from langsmith import traceable

log = logging.getLogger(__name__)


class RetrievedChunk(TypedDict):
    chunk_id: int
    content: str
    page_num: int
    filename: str
    score: float  # 1 - cosine_distance, so higher = more similar


def vector_to_pg(v: list[float]) -> str:
    """Format a Python list as a pgvector literal."""
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"

@traceable(name="retrieve", run_type="retriever")
def retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Retrieve the top-K chunks most semantically similar to `question`.
    
    Returns chunks with metadata (filename, page) and a similarity score.
    """
    # Step 1: embed the question
    query_vec = embed_text(question)
    pg_vec = vector_to_pg(query_vec)
    
    # Step 2: nearest-neighbor search in Postgres
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.content,
                    c.page_num,
                    d.filename,
                    1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (pg_vec, pg_vec, top_k),
            )
            rows = cur.fetchall()
    
    return [
        {
            "chunk_id": row[0],
            "content": row[1],
            "page_num": row[2],
            "filename": row[3],
            "score": float(row[4]),
        }
        for row in rows
    ]