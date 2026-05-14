"""Build a grounded answer from retrieved chunks + question."""
import logging
from src.rag.aws.bedrock import claude_invoke, HAIKU
from src.rag.retrieval.retriever import RetrievedChunk
from langsmith import traceable

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a research assistant that answers questions \
about academic papers. You will be given a USER QUESTION and a set of \
CONTEXT PASSAGES extracted from research papers.

Rules you MUST follow:
1. Answer ONLY using information from the CONTEXT PASSAGES.
2. If the context does not contain enough information, say so clearly. \
Do not guess or invent facts.
3. Cite the source for every claim using the format [source_N] where N \
is the passage number.
4. Be concise. 3-5 sentences unless the question requires more detail.
5. If the question is conversational or off-topic, politely say you only \
answer questions about the provided documents."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format chunks into a numbered context block for the prompt."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[source_{i}] (from {c['filename']}, page {c['page_num']})\n"
            f"{c['content']}\n"
        )
    return "\n---\n".join(parts)

@traceable(name="synthesize", run_type="chain")
def synthesize_answer(
    question: str,
    chunks: list[RetrievedChunk],
    model_id: str = HAIKU,
) -> str:
    """
    Generate an answer to `question` grounded in `chunks`.
    """
    if not chunks:
        return "I couldn't find any relevant information in the documents."
    
    context = build_context_block(chunks)
    
    prompt = f"""CONTEXT PASSAGES:
{context}

USER QUESTION:
{question}

Answer the question using only the context above. Cite sources as [source_N]."""
    
    return claude_invoke(
        prompt=prompt,
        system=SYSTEM_PROMPT,
        model_id=model_id,
        temperature=0.0,
        max_tokens=600,
    )