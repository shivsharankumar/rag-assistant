"""
Ingest a PDF: extract text, chunk it, store in Postgres.

Usage:
    uv run python scripts/ingest.py path/to/paper.pdf
    uv run python scripts/ingest.py --dir ~/rag-data/papers
"""
import argparse
import logging
from pathlib import Path
import sys
# from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.ingestion.extract import extract_text
from src.rag.ingestion.chunker import token_chunk
from src.rag.db.connection import get_connection


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