"""Upload eval dataset to LangSmith for run tracking."""
import json
from pathlib import Path
from langsmith import Client


DATASET_NAME = "rag-research-papers-v1"


def main():
    client = Client()
    
    # Create dataset if missing
    try:
        ds = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset exists: {ds.id}")
    except Exception:
        ds = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Research papers RAG eval set with ground truths",
        )
        print(f"Created dataset: {ds.id}")
    
    # Upload examples
    items = [json.loads(line) for line in Path("eval/dataset.jsonl").read_text().splitlines()]
    in_scope = [i for i in items if i.get("ground_truth")]
    
    for item in in_scope:
        client.create_example(
            dataset_id=ds.id,
            inputs={"question": item["question"]},
            outputs={
                "ground_truth": item["ground_truth"],
                "expected_source": item["expected_source"],
            },
        )
    print(f"Uploaded {len(in_scope)} examples")


if __name__ == "__main__":
    main()