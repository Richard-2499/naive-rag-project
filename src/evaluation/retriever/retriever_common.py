import json
from pathlib import Path

from src.evaluation.dataset.retriever_dataset import RetrieverEvalDataset, EvaluationCase
from config.config_loader import load_config
from src.evaluation.retriever.retriever_evaluator import RetrievalEvaluator
from src.evaluation.metrics.metrics import Metrics

config = load_config()
dataset_path = config["paths"]["retriever_eval_dataset"]

def load_dataset() -> list[EvaluationCase]:
    """
    加载评估数据集
    Returns:
        RetrieverEvalDataset: 评估数据集
    """
    return RetrieverEvalDataset(dataset_path).load()


def run_evaluation(retriever, output_path: str, top_k: int = 15):

    dataset = load_dataset()
    evaluator = RetrievalEvaluator(retriever = retriever, metrics = Metrics())
    results = evaluator.evaluate(dataset = dataset, top_k = top_k)

    summary = {
        "details": results,
        "summary": {
            "total_queries": len(results),
            "avg_recall": sum(item["recall"] for item in results) / len(results),
            "avg_mrr": sum(item["mrr"] for item in results) / len(results)
        }
    }
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

    print(f"Evaluation Result saved to {output_path}:")