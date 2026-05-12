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