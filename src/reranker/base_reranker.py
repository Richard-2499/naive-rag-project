from abc import ABC, abstractmethod
from src.schema.retrieval_result import RetrievalResult

class BaseReranker(ABC):
    """
    Reranker 统一接口：
        所有reranker实现必须遵守该接口
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ):
        """
        对candidate结果重新排序
        Args:
            query:
                用户问题

            results:
                Retriever召回结果

            top_k:
                最终保留数量

        Returns:
            rerank后的结果
        """
        pass