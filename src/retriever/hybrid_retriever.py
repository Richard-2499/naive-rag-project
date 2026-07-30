from src.schema.retrieval_result import RetrievalResult


class HybridRetriever:
    """
    混合检索器，负责调用retrievers 并返回结果，不负责回答
    """
    def __init__(self, bm25_retriever, vector_retriever, fusion_strategy, candidate_k = 20):
        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vector_retriever
        self._fusion_strategy = fusion_strategy
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        # 1. vector search
        vector_results = self._vector_retriever.retrieve(query, self._candidate_k)
        # 2. bm25 search
        bm25_results = self._bm25_retriever.retrieve(query, self._candidate_k)
        # 3. fusion
        fused_results = self._fusion_strategy.fuse(bm25_results, vector_results, top_k)
        # 4. return top-k
        return fused_results