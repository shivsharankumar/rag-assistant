"""Full RAGAS-style evaluation across the eval set, comparing configurations."""
import json
import logging
import statistics
from pathlib import Path
from dataclasses import dataclass
import sys
# from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.retrieval.hybrid import hybrid_retrieve
from src.rag.retrieval.reranker import rerank
from src.rag.synthesis.answer import synthesize_answer
from eval.metrics import faithfulness, answer_relevance, context_precision


logging.basicConfig(level=logging.WARNING)  # silence noisy logs for clean output


@dataclass
class EvalResult:
    question: str
    answer: str
    faithfulness: float
    relevance: float
    precision: float


def run_pipeline(question: str, use_reranker: bool) -> tuple[str, list[dict]]:
    """Run the RAG pipeline with optional reranking. Returns (answer, chunks)."""
    candidates = hybrid_retrieve(question, top_k=20, fetch_k=20)
    if use_reranker:
        chunks = rerank(question, candidates, top_k=5)
    else:
        chunks = candidates[:5]
    answer = synthesize_answer(question, chunks)
    return answer, chunks


def evaluate_config(dataset: list[dict], use_reranker: bool, label: str):
    print(f"\n{'='*70}")
    print(f"Configuration: {label}")
    print('='*70)
    
    in_scope = [d for d in dataset if d.get("expected_source")]
    results = []
    
    for i, item in enumerate(in_scope, 1):
        q = item["question"]
        print(f"  [{i}/{len(in_scope)}] {q[:60]}...")
        
        answer, chunks = run_pipeline(q, use_reranker=use_reranker)
        
        f_score, _ = faithfulness(answer, chunks)
        r_score, _ = answer_relevance(q, answer)
        p_score, _ = context_precision(q, chunks)
        
        results.append(EvalResult(
            question=q, answer=answer,
            faithfulness=f_score, relevance=r_score, precision=p_score,
        ))
        print(f"      F={f_score:.2f}  R={r_score:.2f}  P={p_score:.2f}")
    
    f_avg = statistics.mean(r.faithfulness for r in results)
    r_avg = statistics.mean(r.relevance for r in results)
    p_avg = statistics.mean(r.precision for r in results)
    
    print(f"\n  AVG  Faithfulness: {f_avg:.3f}")
    print(f"  AVG  Answer Relevance: {r_avg:.3f}")
    print(f"  AVG  Context Precision: {p_avg:.3f}")
    
    return {"faithfulness": f_avg, "relevance": r_avg, "precision": p_avg}


def main():
    dataset = [json.loads(line) for line in Path("eval/dataset.jsonl").read_text().splitlines()]
    
    baseline = evaluate_config(dataset, use_reranker=False, label="BASELINE (no reranker)")
    reranked = evaluate_config(dataset, use_reranker=True, label="WITH RERANKER")
    
    print("\n" + "="*70)
    print("LIFT FROM RERANKING")
    print("="*70)
    for metric in ("faithfulness", "relevance", "precision"):
        delta = reranked[metric] - baseline[metric]
        pct = (delta / baseline[metric] * 100) if baseline[metric] else 0
        print(f"  {metric:20s}  {baseline[metric]:.3f} → {reranked[metric]:.3f}  ({pct:+.1f}%)")


if __name__ == "__main__":
    main()