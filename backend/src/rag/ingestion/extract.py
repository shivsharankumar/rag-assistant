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
        text = page.get_text().replace("\x00", "")
        # Skip empty pages (cover pages, blanks)
        if text.strip():
            pages.append({"page_num": i + 1, "text": text})
    
    return {
        "filename": pdf_path.name,
        "total_pages": len(doc),
        "pages": pages,
    }