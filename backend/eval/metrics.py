"""LLM-as-judge eval metrics, in the style of RAGAS."""
import json
import re
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.aws.bedrock import claude_invoke, SONNET


log = logging.getLogger(__name__)


def _extract_score(response: str, default: float = 0.5) -> tuple[float, str]:
    """Extract a 0-1 score and reasoning from an LLM judgment response."""
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return default, "parse_error"
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score", default))
        reasoning = str(data.get("reasoning", ""))
        return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, ValueError):
        return default, "parse_error"


# ---------- Faithfulness ----------

FAITHFULNESS_SYSTEM = """You evaluate whether an AI-generated answer is \
faithful to a set of source passages.

Faithfulness = every claim in the answer is supported by the passages.

Score from 0 to 1:
- 1.0 = every claim is directly supported by the passages
- 0.7-0.9 = most claims supported, minor unsupported inference
- 0.4-0.6 = mix of supported and unsupported claims
- 0.1-0.3 = mostly hallucinated
- 0.0 = entirely fabricated

Output ONLY a JSON object: {"score": <0-1>, "reasoning": "<brief>"}"""


def faithfulness(answer: str, chunks: list[dict]) -> tuple[float, str]:
    """Score 0-1: how well is the answer supported by retrieved chunks?"""
    if not answer or not chunks:
        return 0.0, "empty_input"
    
    context = "\n---\n".join(
        f"PASSAGE {i+1}: {c['content']}"
        for i, c in enumerate(chunks)
    )
    prompt = f"PASSAGES:\n{context}\n\nANSWER:\n{answer}\n\nEvaluate faithfulness."
    
    response = claude_invoke(
        prompt=prompt, system=FAITHFULNESS_SYSTEM,
        model_id=SONNET, max_tokens=300, temperature=0.0,
    )
    return _extract_score(response)


# ---------- Answer Relevance ----------

RELEVANCE_SYSTEM = """You evaluate whether an answer actually addresses \
the user's question.

Relevance = the answer is on-topic and answers the question asked, \
not a different question.

Score from 0 to 1:
- 1.0 = directly and completely answers the question
- 0.7-0.9 = answers the core question with minor digression
- 0.4-0.6 = partially answers or answers a related question
- 0.1-0.3 = mostly off-topic
- 0.0 = doesn't answer at all

Output ONLY a JSON object: {"score": <0-1>, "reasoning": "<brief>"}"""


def answer_relevance(question: str, answer: str) -> tuple[float, str]:
    """Score 0-1: does the answer address the question?"""
    if not answer:
        return 0.0, "empty_answer"
    
    prompt = f"QUESTION: {question}\n\nANSWER: {answer}\n\nEvaluate relevance."
    response = claude_invoke(
        prompt=prompt, system=RELEVANCE_SYSTEM,
        model_id=SONNET, max_tokens=200, temperature=0.0,
    )
    return _extract_score(response)


# ---------- Context Precision ----------

PRECISION_SYSTEM = """You evaluate whether retrieved passages are \
relevant to a user's question.

For each passage, judge whether it's useful for answering the question.

Context precision = (number of relevant passages) / (total passages).

Output ONLY a JSON object: 
{
  "relevant": [true/false, true/false, ...],
  "reasoning": "<brief explanation>"
}

The 'relevant' array length must equal the number of passages."""


def context_precision(question: str, chunks: list[dict]) -> tuple[float, str]:
    """Fraction of retrieved chunks that are relevant to the question."""
    if not chunks:
        return 0.0, "no_chunks"
    
    passages = "\n---\n".join(
        f"PASSAGE {i+1}: {c['content']}"
        for i, c in enumerate(chunks)
    )
    prompt = f"QUESTION: {question}\n\nPASSAGES:\n{passages}"
    
    response = claude_invoke(
        prompt=prompt, system=PRECISION_SYSTEM,
        model_id=SONNET, max_tokens=300, temperature=0.0,
    )
    
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return 0.5, "parse_error"
    try:
        data = json.loads(match.group(0))
        relevant = data.get("relevant", [])
        if not relevant:
            return 0.0, "no_judgments"
        precision = sum(1 for r in relevant if r) / len(relevant)
        return precision, str(data.get("reasoning", ""))
    except (json.JSONDecodeError, ValueError, ZeroDivisionError):
        return 0.5, "parse_error"