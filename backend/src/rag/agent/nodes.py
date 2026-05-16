"""Individual nodes for the RAG agent graph."""
import logging
import json
from src.rag.agent.state import AgentState
from src.rag.aws.bedrock import claude_invoke, HAIKU
from src.rag.retrieval.hybrid import hybrid_retrieve
from src.rag.retrieval.reranker import rerank as rerank_fn
from src.rag.aws.bedrock import SONNET
log = logging.getLogger(__name__)
import re

# ---------- Router ----------

ROUTER_SYSTEM = """You classify user questions about research papers into one of:
- factual: a direct lookup (e.g., "What is multi-head attention?")
- comparison: comparing two or more concepts/papers (e.g., "How does BERT differ from GPT?")
- summary: asking for a synthesis or overview (e.g., "Summarize the key findings")
- out_of_scope: unrelated to research papers (greetings, weather, cooking, etc.)

Respond with ONLY one word: factual, comparison, summary, or out_of_scope.Do not include <reasoning> tags, internal monologue, or any meta-commentary. 
Output only the requested content directly."""




def reranker_node(state: AgentState) -> AgentState:
    """Rerank the retrieved chunks using LLM scoring."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"trace": state.get("trace", []) + ["reranker:skip_empty"]}
    
    q = state.get("rewritten_question") or state["question"]
    reranked = rerank_fn(q, chunks, top_k=5)
    
    log.info(
        f"[reranker] Reranked {len(chunks)} → {len(reranked)}, "
        f"top score: {reranked[0].get('rerank_score', 0):.1f}"
    )
    return {
        "chunks": reranked,
        "trace": state.get("trace", []) + [f"reranker:{len(reranked)}"],
    }

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
    q = state.get("rewritten_question") or state["question"]
    # Fetch wider — reranker will narrow this to 5
    chunks = hybrid_retrieve(q, top_k=20, fetch_k=20)
    log.info(f"[retriever] Got {len(chunks)} chunks")
    return {
        "chunks": chunks,
        "trace": state.get("trace", []) + [f"retriever:{len(chunks)}"],
    }


# ---------- Grounding check ----------

GROUNDING_MIN_SCORE = 0.35  # tune based on eval data
GROUNDING_MIN_CHUNKS = 1
GROUNDING_MIN_RERANK = 4.0  # on 0-10 scale, 4 = "tangentially related"


def grounding_check(state: AgentState) -> AgentState:
    chunks = state.get("chunks", [])
    
    if not chunks:
        return {
            "is_grounded": False,
            "grounding_reason": "no_chunks",
            "trace": state.get("trace", []) + ["grounding:fail_no_chunks"],
        }
    
    # Use rerank score if available (more accurate than RRF for grounding)
    top_score = max(c.get("rerank_score", 0) for c in chunks)
    if top_score < GROUNDING_MIN_RERANK:
        return {
            "is_grounded": False,
            "grounding_reason": f"low_rerank_score:{top_score:.1f}",
            "trace": state.get("trace", []) + [f"grounding:fail_low_score_{top_score:.1f}"],
        }
    
    return {
        "is_grounded": True,
        "trace": state.get("trace", []) + [f"grounding:pass_{top_score:.1f}"],
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




CRITIC_SYSTEM = """You are a strict fact-checker evaluating an AI-generated \
answer to a user question, given the source passages it was supposed to use.

Your job:
1. Identify every distinct factual claim in the ANSWER.
2. For each claim, check whether it is supported by the PASSAGES.
3. Score overall faithfulness from 0-10:
   - 10 = every claim is directly supported
   - 7-9 = mostly supported, minor inference
   - 4-6 = mix of supported and unsupported claims
   - 1-3 = mostly unsupported, hallucinated
   - 0 = entirely fabricated

Output ONLY a JSON object in this exact format:
{
  "score": <0-10 number>,
  "reasoning": "<2-3 sentence explanation>",
  "unsupported_claims": ["<claim 1>", "<claim 2>", ...]
}

If all claims are supported, "unsupported_claims" should be an empty array."""


def _parse_critic_response(response: str) -> dict:
    """Defensive parsing of critic JSON output."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        log.warning(f"No JSON in critic response: {response[:200]}")
        return {"score": 5.0, "reasoning": "parse_error", "unsupported_claims": []}
    try:
        data = json.loads(match.group(0))
        return {
            "score": float(data.get("score", 5.0)),
            "reasoning": str(data.get("reasoning", "")),
            "unsupported_claims": list(data.get("unsupported_claims", [])),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        log.warning(f"Critic parse error: {e}")
        return {"score": 5.0, "reasoning": "parse_error", "unsupported_claims": []}


def critic(state: AgentState) -> AgentState:
    """Evaluate whether the answer is grounded in the retrieved chunks."""
    answer = state.get("answer", "")
    chunks = state.get("chunks", [])
    
    if not answer or not chunks:
        return {
            "critic_score": 0.0,
            "critic_reasoning": "no_answer_or_chunks",
            "critic_unsupported_claims": [],
            "trace": state.get("trace", []) + ["critic:skip"],
        }
    
    context = "\n---\n".join(
        f"PASSAGE {i+1}: {c['content']}"
        for i, c in enumerate(chunks)
    )
    prompt = f"""QUESTION: {state['question']}

    PASSAGES:
    {context}

    ANSWER:
    {answer}

    Evaluate the answer's faithfulness to the passages."""
    
    response = claude_invoke(
        prompt=prompt,
        system=CRITIC_SYSTEM,
        model_id=SONNET,         # stronger judge model
        max_tokens=500,
        temperature=0.0,
    )
    
    parsed = _parse_critic_response(response)
    
    log.info(
        f"[critic] Score: {parsed['score']:.1f}/10, "
        f"unsupported claims: {len(parsed['unsupported_claims'])}"
    )
    
    return {
        "critic_score": parsed["score"],
        "critic_reasoning": parsed["reasoning"],
        "critic_unsupported_claims": parsed["unsupported_claims"],
        "trace": state.get("trace", []) + [f"critic:{parsed['score']:.1f}"],
    }