"""Compare baseline (Week 2) vs agent (Week 3) on the eval set."""
import json
from pathlib import Path
# from pathlib import Path
import sys
# from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.retrieval.retriever import retrieve as retrieve_baseline
from src.rag.retrieval.hybrid import hybrid_retrieve
from src.rag.agent.graph import agent_graph


def main():
    dataset = [json.loads(line) for line in Path("eval/dataset.jsonl").read_text().splitlines()]
    
    # Metric 1: top-1 retrieval accuracy (baseline vector vs hybrid)
    baseline_correct = 0
    hybrid_correct = 0
    in_scope = [d for d in dataset if d["expected_source"]]
    
    for item in in_scope:
        baseline_top = retrieve_baseline(item["question"], top_k=1)
        hybrid_top = hybrid_retrieve(item["question"], top_k=1)
        if baseline_top and baseline_top[0]["filename"] == item["expected_source"]:
            baseline_correct += 1
        if hybrid_top and hybrid_top[0]["filename"] == item["expected_source"]:
            hybrid_correct += 1
    
    print(f"\n=== Retrieval top-1 accuracy (in-scope only) ===")
    print(f"Baseline (pure vector): {baseline_correct}/{len(in_scope)} = {baseline_correct/len(in_scope):.0%}")
    print(f"Hybrid (vector + BM25): {hybrid_correct}/{len(in_scope)} = {hybrid_correct/len(in_scope):.0%}")
    
    # Metric 2: router accuracy
    router_correct = 0
    for item in dataset:
        result = agent_graph.invoke({"question": item["question"]})
        predicted_type = result.get("question_type", "unknown")
        expected_type = item["type"]
        is_correct = predicted_type == expected_type
        router_correct += is_correct
        symbol = "✅" if is_correct else "❌"
        print(f"{symbol} {item['type']:15} | {predicted_type:15} | {item['question'][:50]}")
    
    print(f"\n=== Router accuracy ===")
    print(f"{router_correct}/{len(dataset)} = {router_correct/len(dataset):.0%}")
    
    # Metric 3: out-of-scope handling (did we correctly NOT hallucinate?)
    oos_correct = 0
    oos_items = [d for d in dataset if d["type"] == "out_of_scope"]
    for item in oos_items:
        result = agent_graph.invoke({"question": item["question"]})
        # Pass = either router caught it OR grounding check caught it
        traced = result.get("trace", [])
        if any("apologize" in t for t in traced):
            oos_correct += 1
    
    print(f"\n=== Out-of-scope correctly refused ===")
    print(f"{oos_correct}/{len(oos_items)} = {oos_correct/len(oos_items):.0%}")


if __name__ == "__main__":
    main()