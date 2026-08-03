
"""
输出评测结果， 例如
    Evaluation Result
        Dataset:  100 queries
        Recall@5:  0.86
        MRR:   0.72

方便比较： 模型， 参数， 优化前后结果

"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class EvaluationReport:
    @staticmethod
    def summarize(results: list[dict[str, Any]]):
        total = len(results)
        avg_recall = sum(item["recall"] for item in results) / total
        avg_mrr = sum(item["mrr"] for item in results) / total

        return {
            "total_queries": total,
            "avg_recall": avg_recall,
            "avg_mrr": avg_mrr
        }
    @staticmethod
    def print_report(summary):
        print("=" * 80)
        print("Evaluation Result")
        print("=" * 80)

        print(f"Queries: {summary["total_queries"]}")
        print(f"Recall: {summary["avg_recall"]:.3f}")
        print(f"MRR: {summary["avg_mrr"]:.3f}")

    @staticmethod
    def save(result: dict, output_file: str):
        path = Path(output_file)
        report = {
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "result": result
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)