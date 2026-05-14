"""
Generate embeddings for all chunks that don't have one yet.

Idempotent: re-running only embeds chunks where embedding IS NULL.

Usage:
    uv run python scripts/embed_chunks.py
    uv run python scripts/embed_chunks.py --reembed-all   (force re-embed)
"""
import argparse
import logging
import time
from pathlib import Path
import sys
# from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.aws.bedrock import embed_text, EMBEDDING_DIM
from src.rag.db.connection import get_connection


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def fetch_chunks_to_embed(reembed_all: bool = False) -> list[tuple[int, str]]:
    """Return list of (chunk_id, content) needing embeddings."""
    query = "SELECT id, content FROM chunks"
    if not reembed_all:
        query += " WHERE embedding IS NULL"
    query += " ORDER BY id"
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()


def save_embedding(chunk_id: int, embedding: list[float]) -> None:
    """Store an embedding for a chunk."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # pgvector accepts a string-formatted vector like '[0.1,0.2,...]'
            vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
            cur.execute(
                "UPDATE chunks SET embedding = %s::vector WHERE id = %s",
                (vec_str, chunk_id),
            )
        conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reembed-all", action="store_true",
                        help="Re-embed even chunks that already have embeddings")
    args = parser.parse_args()
    
    chunks = fetch_chunks_to_embed(args.reembed_all)
    log.info(f"Embedding {len(chunks)} chunks...")
    
    start = time.time()
    for i, (chunk_id, content) in enumerate(chunks, 1):
        emb = embed_text(content)
        save_embedding(chunk_id, emb)
        
        if i % 25 == 0 or i == len(chunks):
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (len(chunks) - i) / rate if rate > 0 else 0
            log.info(f"  {i}/{len(chunks)} chunks  ({rate:.1f}/s, ETA {eta:.0f}s)")
    
    log.info(f"✅ Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()