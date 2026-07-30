
"""
负责 整个评测流程：
    读取 dataset → 调用 Retriever → 拿 RetrievalResult → 调用 Metrics → 生成结果


"""
import logging
from typing import Any

from src.evaluation.dataset import EvaluationSample



class RetrievalEvaluator:
    """
    RAG 检索评估器， 对给定的检索器在数据集上进行评估，计算各项指标。
    """
    def __init__(self, retriever, metrics):
        self._retriever = retriever
        self._metrics = metrics
        self._logger = logging.getLogger(self.__class__.__name__)

    def evaluate(
            self,
            dataset: list[EvaluationSample],
            top_k: int = 15,
            verbose: bool = True
    ) -> list[dict[str, Any]]:
        '''
        评估检索器在数据集上的表现

        Args:
            dataset: 评估数据集，每个样本包含问题和相关文档 ID
            top_k: 评估时使用的检索数量（Recall@K 中的 K）
            verbose: 是否打印进度信息

        Returns:
            包含以下内容的字典：
            - "results": 每个样本的详细评估结果列表
            - "summary": 汇总统计（平均 recall, 平均 mrr 等）

        Example:
            evaluator = RetrievalEvaluator(retriever, metrics)
            results = evaluator.evaluate(dataset, top_k=5)
            print(results["summary"]["avg_recall"])
            0.78
        '''

        results = []
        for sample in dataset:
            retrieved_results = self._retriever.retrieve(sample.query, top_k=top_k)
            retrieved_ids = [item.chunk_id for item in retrieved_results]
            recall = (self._metrics.recall_at_k(retrieved_ids, sample.relevant_chunk_ids, top_k))
            mrr = (self._metrics.mrr(retrieved_ids, sample.relevant_chunk_ids))
            results.append({
                "query": sample.query,
                "relevant_chunk_ids": sample.relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "recall": recall,
                "mrr": mrr
            })
        return results