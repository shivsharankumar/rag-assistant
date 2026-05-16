"""LLM-based listwise reranker for retrieval results."""
import json
import re
import logging
from langsmith import traceable
from src.rag.aws.bedrock import claude_invoke, HAIKU


log = logging.getLogger(__name__)


RERANKER_SYSTEM = """You are a reranker that scores how well each passage \
answers a user's question.

You will be given a QUESTION and a numbered list of PASSAGES. For each passage, \
score from 0 to 10:
- 10 = directly and completely answers the question
- 7-9 = highly relevant, contains key information
- 4-6 = tangentially related, partial context
- 1-3 = topically related but doesn't help answer
- 0 = unrelated

Output ONLY a JSON object in this exact format, nothing else:
{"scores": [score_for_passage_1, score_for_passage_2, ...]}

The array length must equal the number of passages."""


def _build_rerank_prompt(question: str, chunks: list[dict]) -> str:
    passages_text = "\n\n".join(
        f"PASSAGE {i+1}:\n{c['content']}"
        for i, c in enumerate(chunks)
    )
    return f"QUESTION: {question}\n\n{passages_text}"


def _parse_scores(response: str, expected_count: int) -> list[float]:
    """Extract scores list from LLM response. Defensive parsing."""
    # Find first JSON object in response
    match = re.search(r"\{[^{}]*\"scores\"[^{}]*\}", response, re.DOTALL)
    if not match:
        log.warning(f"No JSON found in reranker response: {response[:200]}")
        return [5.0] * expected_count  # neutral fallback
    try:
        data = json.loads(match.group(0))
        scores = data.get("scores", [])
        if len(scores) != expected_count:
            log.warning(f"Expected {expected_count} scores, got {len(scores)}")
            # Pad or truncate defensively
            scores = (scores + [5.0] * expected_count)[:expected_count]
        return [float(s) for s in scores]
    except (json.JSONDecodeError, ValueError) as e:
        log.warning(f"Failed to parse scores: {e}")
        return [5.0] * expected_count


@traceable(name="rerank", run_type="chain")
def rerank(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank chunks by LLM-judged relevance to the question.
    
    Args:
        question: User question
        chunks: Candidates from hybrid_retrieve (list of dicts with at least 'content')
        top_k: How many to return after reranking
    
    Returns:
        Top-K chunks sorted by rerank_score (descending), with rerank_score 
        added to each dict.
    """
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        # Nothing to rerank — just score for transparency
        log.info(f"[rerank] {len(chunks)} chunks <= top_k={top_k}, scoring only")
    
    prompt = _build_rerank_prompt(question, chunks)
    response = claude_invoke(
        prompt=prompt,
        system=RERANKER_SYSTEM,
        model_id=HAIKU,
        max_tokens=200,
        temperature=0.0,
    )
    
    scores = _parse_scores(response, len(chunks))
    
    # Attach scores and sort
    scored = [
        {**chunk, "rerank_score": score}
        for chunk, score in zip(chunks, scores)
    ]
    scored.sort(key=lambda x: -x["rerank_score"])
    
    log.info(
        f"[rerank] Scored {len(chunks)} chunks. "
        f"Top score: {scored[0]['rerank_score']:.1f}, "
        f"K-th score: {scored[min(top_k-1, len(scored)-1)]['rerank_score']:.1f}"
    )
    return scored[:top_k]