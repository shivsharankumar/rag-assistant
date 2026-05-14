"""Hybrid retrieval: vector + lexical, fused with Reciprocal Rank Fusion."""
import logging
from typing import TypedDict
from langsmith import traceable
from src.rag.aws.bedrock import embed_text
from src.rag.db.connection import get_connection


log = logging.getLogger(__name__)


class RetrievedChunk(TypedDict):
    chunk_id: int
    content: str
    page_num: int
    filename: str
    vector_rank: int | None
    lexical_rank: int | None
    rrf_score: float


def vector_to_pg(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _vector_search(question: str, top_k: int) -> list[dict]:
    """Top-K by cosine similarity."""
    query_vec = embed_text(question)
    pg_vec = vector_to_pg(query_vec)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.content, c.page_num, d.filename,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (pg_vec, pg_vec, top_k),
            )
            return [
                {"chunk_id": r[0], "content": r[1], "page_num": r[2],
                 "filename": r[3], "score": float(r[4])}
                for r in cur.fetchall()
            ]


def _lexical_search(question: str, top_k: int) -> list[dict]:
    """Top-K by Postgres full-text rank."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.content, c.page_num, d.filename,
                       ts_rank(c.content_tsv, query) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id,
                     plainto_tsquery('english', %s) query
                WHERE c.content_tsv @@ query
                ORDER BY score DESC
                LIMIT %s
                """,
                (question, top_k),
            )
            return [
                {"chunk_id": r[0], "content": r[1], "page_num": r[2],
                 "filename": r[3], "score": float(r[4])}
                for r in cur.fetchall()
            ]


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 60,
) -> dict[int, dict]:
    """
    Combine multiple ranked lists using RRF.
    Returns dict of chunk_id -> fused result with rrf_score.
    
    k=60 is the standard constant from the original RRF paper.
    """
    fused: dict[int, dict] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            cid = doc["chunk_id"]
            if cid not in fused:
                fused[cid] = {**doc, "rrf_score": 0.0,
                              "vector_rank": None, "lexical_rank": None}
            fused[cid]["rrf_score"] += 1.0 / (k + rank)
    return fused


@traceable(name="hybrid_retrieve", run_type="retriever")
def hybrid_retrieve(question: str, top_k: int = 5, fetch_k: int = 20) -> list[RetrievedChunk]:
    """
    Retrieve top-K chunks using hybrid search:
      1. Get top fetch_k from vector search
      2. Get top fetch_k from lexical search
      3. Fuse with RRF
      4. Return top_k after fusion
    """
    vec_results = _vector_search(question, fetch_k)
    lex_results = _lexical_search(question, fetch_k)
    
    # Tag ranks before fusion
    vec_ranked = [{**r, "vector_rank": i+1} for i, r in enumerate(vec_results)]
    lex_ranked = [{**r, "lexical_rank": i+1} for i, r in enumerate(lex_results)]
    
    fused = reciprocal_rank_fusion([vec_ranked, lex_ranked])
    
    # Re-attach the per-source ranks
    for r in vec_ranked:
        fused[r["chunk_id"]]["vector_rank"] = r["vector_rank"]
    for r in lex_ranked:
        fused[r["chunk_id"]]["lexical_rank"] = r["lexical_rank"]
    
    sorted_results = sorted(fused.values(), key=lambda x: -x["rrf_score"])[:top_k]
    
    log.info(
        f"Hybrid retrieve: {len(vec_results)} vector + {len(lex_results)} lexical "
        f"→ {len(fused)} unique → top {top_k}"
    )
    return sorted_results