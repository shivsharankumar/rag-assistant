"""Sanity check: does the top retrieved source match expected?"""
import json
from pathlib import Path
import sys
# from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.retrieval.retriever import retrieve


def main():
    dataset = [json.loads(line) for line in Path("eval/dataset.jsonl").read_text().splitlines()]
    
    correct = 0
    for item in dataset:
        chunks = retrieve(item["question"], top_k=1)
        top_source = chunks[0]["filename"] if chunks else None
        is_correct = top_source == item["expected_source"]
        correct += is_correct
        symbol = "✅" if is_correct else "❌"
        print(f"{symbol} Q: {item['question'][:60]}...")
        print(f"   Expected: {item['expected_source']}")
        print(f"   Got:      {top_source}")
    
    print(f"\nTop-1 retrieval accuracy: {correct}/{len(dataset)} = {correct/len(dataset):.0%}")


if __name__ == "__main__":
    main()