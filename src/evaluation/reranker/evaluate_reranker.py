from typing import Protocol

import numpy as np

from src.evaluation.dataset.reranker_dataset import RerankerEvalDataset
from src.evaluation.metrics.metrics import Metrics
from src.reranker.bge_reranker import BGEReranker
from src.retriever.hybrid_retriever import HybridRetriever
from src.schema.retrieval_result import RetrievalResult


# class RetrievalProtocol(Protocol):
#     """
#     Retriever接口定义。
#     Evaluation只依赖retrieve能力，
#     不关心具体是VectorRetriever、BM25Retriever还是HybridRetriever。
#     """
#     def retrieve(self, query: str, top_k: int = 15) -> list[RetrievalResult]:
#         ...
#
# class RerankerProtocol(Protocol):
#     """
#     Reranker接口定义。
#     Evaluation只依赖rerank能力，
#     可以替换BGE、Cohere等不同reranker实现。
#     """
#     def rerank(self, query: str, retrieval_results: list[RetrievalResult]) -> list:
#         ...

class RerankerEvaluator:

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BGEReranker,
        dataset: RerankerEvalDataset,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.dataset: RerankerEvalDataset = dataset

    def evaluate(self, candidate_k: int, rerank_k: int):
        """
        评估Reranker的效果，同时返回rerank前后的指标对比
        Returns:
            dict: 包含rerank前后对比的评估结果
        """
        # 保存 rerank 前后评估结果
        before_rerank_results: dict[str, list[float]] = {
            "ndcg_scores": [],
            "mrr_scores": []
        }
        after_rerank_results: dict[str, list[float]] = {
            "ndcg_scores": [],
            "mrr_scores": []
        }
        for sample in self.dataset.samples:
            # 第一步：Retriever负责召回候选集
            # 这里评价的是Reranker，所以需要固定candidate_k
            candidates = self.retriever.retrieve(
                query=sample.query,
                top_k=candidate_k
            )
            # 第二步：Reranker只负责candidate集合内部重新排序
            reranked_results = self.reranker.rerank(
                query=sample.query,
                results=candidates,
                top_k=rerank_k
            )
            # reranked_chunk_ids是Reranker实际输出的排序结果。
            # Evaluation通过比较它和ground_truth_relevance判断排序质量。
            # 注意：candidates可能包含超过rerank_k个结果，需要截断到相同长度进行比较
            ground_truth_ids: list[str] = [
                i for i, _ in sample.relevant_chunks.items()
            ][:rerank_k]
            candidate_chunk_ids: list[str] = [
                item.chunk_id
                for item in candidates[:rerank_k]
            ]
            reranked_chunk_ids: list[str] = [
                item.chunk_id
                for item in reranked_results[:rerank_k]
            ]

            # 评估 rerank 前（baseline）
            # 计算 NDCG
            before_ndcg_score = Metrics.ndcg_at_k(
                reranked_chunk_ids=candidate_chunk_ids,
                ground_truth=sample.relevant_chunks,
                top_k=rerank_k
            )
            # 计算MRR：衡量第一个相关chunk是否排在前面
            before_mrr_score = Metrics.mrr(
                retrieved_ids=candidate_chunk_ids,
                relevant_ids = ground_truth_ids[:rerank_k],
            )

            # 评估 rerank 后
            # 计算NDCG：衡量Reranker排序结果是否接近人工理想排序
            after_ndcg_score = Metrics.ndcg_at_k(
                reranked_chunk_ids=reranked_chunk_ids,
                ground_truth=sample.relevant_chunks,
                top_k=rerank_k
            )

            # 计算MRR：衡量第一个相关chunk是否排在前面
            after_mrr_score = Metrics.mrr(
                retrieved_ids=reranked_chunk_ids,
                relevant_ids = ground_truth_ids[:rerank_k],
            )
            before_rerank_results["ndcg_scores"].append(before_ndcg_score)
            before_rerank_results["mrr_scores"].append(before_mrr_score)
            after_rerank_results["ndcg_scores"].append(after_ndcg_score)
            after_rerank_results["mrr_scores"].append(after_mrr_score)

        return self._format_comparison_results(
            before=before_rerank_results,
            after=after_rerank_results
        )

    def _format_comparison_results(self, before: dict, after: dict):

        def calc_avg(scores: list[float]):
            return sum(scores) / len(scores) if scores else 0.0
        # 计算改进幅度
        ndcg_improvement = calc_avg(after["ndcg_scores"]) - calc_avg(before["ndcg_scores"])
        mrr_improvement = calc_avg(after["mrr_scores"]) - calc_avg(before["mrr_scores"])

        return {
            "total_queries": len(before["ndcg_scores"]),
            "ndcg_scores": {
                "before": before["ndcg_scores"],
                "after": after["ndcg_scores"]
            },
            "mrr_scores": {
                "before": before["mrr_scores"],
                "after": after["mrr_scores"]
            },
            "avg_ndcg": {
                "before": calc_avg(before["ndcg_scores"]),
                "after": calc_avg(after["ndcg_scores"]),
                "improvement": ndcg_improvement
            },
            "avg_mrr": {
                "before": calc_avg(before["mrr_scores"]),
                "after": calc_avg(after["mrr_scores"]),
                "improvement": mrr_improvement
            }
        }
