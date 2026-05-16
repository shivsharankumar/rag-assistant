"""Run pipeline against the LangSmith dataset, store as an experiment."""
from langsmith import Client
from langsmith.evaluation import evaluate
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rag.agent.graph import agent_graph
from eval.metrics import faithfulness, answer_relevance


DATASET_NAME = "rag-research-papers-v1"


def target(inputs: dict) -> dict:
    """Run the agent graph on a single example."""
    state = agent_graph.invoke({"question": inputs["question"]})
    return {
        "answer": state.get("answer", ""),
        "chunks": state.get("chunks", []),
        "trace": state.get("trace", []),
    }


def faithfulness_evaluator(run, example):
    answer = run.outputs.get("answer", "")
    chunks = run.outputs.get("chunks", [])
    score, reasoning = faithfulness(answer, chunks)
    return {"key": "faithfulness", "score": score, "comment": reasoning}


def relevance_evaluator(run, example):
    question = example.inputs["question"]
    answer = run.outputs.get("answer", "")
    score, reasoning = answer_relevance(question, answer)
    return {"key": "answer_relevance", "score": score, "comment": reasoning}


def main():
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[faithfulness_evaluator, relevance_evaluator],
        experiment_prefix="rag-week4",
        max_concurrency=2,
    )
    print(results)


if __name__ == "__main__":
    main()