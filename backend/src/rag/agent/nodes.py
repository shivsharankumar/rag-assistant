"""Individual nodes for the RAG agent graph."""
import logging
import json
from src.rag.agent.state import AgentState
from src.rag.aws.bedrock import claude_invoke, HAIKU
from src.rag.retrieval.hybrid import hybrid_retrieve


log = logging.getLogger(__name__)


# ---------- Router ----------

ROUTER_SYSTEM = """You classify user questions about research papers into one of:
- factual: a direct lookup (e.g., "What is multi-head attention?")
- comparison: comparing two or more concepts/papers (e.g., "How does BERT differ from GPT?")
- summary: asking for a synthesis or overview (e.g., "Summarize the key findings")
- out_of_scope: unrelated to research papers (greetings, weather, cooking, etc.)

Respond with ONLY one word: factual, comparison, summary, or out_of_scope.Do not include <reasoning> tags, internal monologue, or any meta-commentary. 
Output only the requested content directly."""


def router(state: AgentState) -> AgentState:
    """Classify the question type."""
    log.info(f"[router] Question: {state['question'][:80]}")
    
    response = claude_invoke(
        prompt=state["question"],
        system=ROUTER_SYSTEM,
        model_id=HAIKU,
        max_tokens=10,
        temperature=0.0,
    ).strip().lower()
    
    # Defensive: clean up if model adds punctuation
    for valid in ("factual", "comparison", "summary", "out_of_scope"):
        if valid in response:
            qt = valid
            break
    else:
        qt = "factual"  # safe default
    
    log.info(f"[router] → {qt}")
    return {
        "question_type": qt,
        "trace": state.get("trace", []) + [f"router:{qt}"],
    }


# ---------- Query rewriter ----------

REWRITER_SYSTEM = """You improve user questions for semantic retrieval against 
research papers.

Rules:
- Expand abbreviations when context makes them clear
- Add relevant technical terms the user might have omitted  
- Keep it as a single question, max 30 words
- Output ONLY the rewritten question on a single line. No preamble, no 
  reasoning tags, no explanation."""


def query_rewriter(state: AgentState) -> AgentState:
    log.info(f"[rewriter] Original: {state['question']}")
    
    rewritten = claude_invoke(
        prompt=state["question"],
        system=REWRITER_SYSTEM,
        model_id=HAIKU,
        max_tokens=80,
        temperature=0.0,
    ).strip().strip('"').strip("'")
    
    # Take only the first non-empty line (defensive)
    lines = [ln.strip() for ln in rewritten.splitlines() if ln.strip()]
    rewritten = lines[0] if lines else state["question"]
    
    # Sanity check: if rewriter produced something weird (too long, empty, 
    # or just keywords), fall back to original
    if len(rewritten) < 5 or len(rewritten) > 300:
        log.warning(f"[rewriter] Bad output ({len(rewritten)} chars), using original")
        rewritten = state["question"]
    
    log.info(f"[rewriter] Rewritten: {rewritten}")
    return {
        "rewritten_question": rewritten,
        "trace": state.get("trace", []) + ["rewriter"],
    }


# ---------- Retriever ----------

def retriever(state: AgentState) -> AgentState:
    """Hybrid retrieval using the rewritten question."""
    q = state.get("rewritten_question") or state["question"]
    chunks = hybrid_retrieve(q, top_k=5, fetch_k=20)
    log.info(f"[retriever] Got {len(chunks)} chunks")
    return {
        "chunks": chunks,
        "trace": state.get("trace", []) + [f"retriever:{len(chunks)}"],
    }


# ---------- Grounding check ----------

GROUNDING_MIN_SCORE = 0.35  # tune based on eval data
GROUNDING_MIN_CHUNKS = 1


def grounding_check(state: AgentState) -> AgentState:
    """Decide whether retrieval is strong enough to synthesize."""
    chunks = state.get("chunks", [])
    
    if len(chunks) < GROUNDING_MIN_CHUNKS:
        return {
            "is_grounded": False,
            "grounding_reason": "no_chunks",
            "trace": state.get("trace", []) + ["grounding:fail_no_chunks"],
        }
    
    # Use RRF score as a proxy — chunks with both vector and lexical hits score higher
    top_score = max(c["rrf_score"] for c in chunks)
    if top_score < (1.0 / (60 + 1)) * 0.6:  # rough threshold; tune later
        return {
            "is_grounded": False,
            "grounding_reason": "low_relevance",
            "trace": state.get("trace", []) + ["grounding:fail_low_score"],
        }
    
    return {
        "is_grounded": True,
        "trace": state.get("trace", []) + ["grounding:pass"],
    }


# ---------- Synthesizer ----------

SYNTH_SYSTEM = """You are a research assistant answering questions from \
academic papers. Follow these rules strictly:
1. Use ONLY the CONTEXT PASSAGES below. Do not use prior knowledge.
2. Cite every claim with [source_N] referring to passage numbers.
3. If the context is insufficient, say so clearly. Do not invent facts.
4. Be concise: 3-5 sentences unless the question demands more."""


def synthesizer(state: AgentState) -> AgentState:
    """Generate the final answer from retrieved chunks."""
    chunks = state["chunks"]
    context = "\n---\n".join(
        f"[source_{i+1}] ({c['filename']}, p{c['page_num']})\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    
    prompt = f"CONTEXT PASSAGES:\n{context}\n\nQUESTION: {state['question']}\n\nAnswer:"
    
    answer = claude_invoke(
        prompt=prompt,
        system=SYNTH_SYSTEM,
        model_id=HAIKU,
        max_tokens=600,
        temperature=0.0,
    )
    return {
        "answer": answer,
        "trace": state.get("trace", []) + ["synthesizer"],
    }


# ---------- Apology nodes (terminal) ----------

def apologize_out_of_scope(state: AgentState) -> AgentState:
    return {
        "answer": "I can only answer questions about the research papers in my knowledge base. Please ask about transformers, BERT, RAG, or related topics.",
        "trace": state.get("trace", []) + ["apologize:out_of_scope"],
    }


def apologize_ungrounded(state: AgentState) -> AgentState:
    return {
        "answer": "I couldn't find sufficient information in my knowledge base to answer that confidently. Try rephrasing or asking about a different aspect.",
        "trace": state.get("trace", []) + ["apologize:ungrounded"],
    }