from abc import ABC, abstractmethod

from src.schema.retrieval_result import RetrievalResult


class FusionStrategy(ABC):
    @abstractmethod
    def fuse(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        融合bm25和向量检索结果

        """
        pass