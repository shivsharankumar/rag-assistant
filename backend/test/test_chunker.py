from src.rag.ingestion.chunker import token_chunk


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