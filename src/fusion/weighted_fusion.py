from src.fusion.fusion_strategy import FusionStrategy
from src.schemas.retrieval_result import RetrievalResult


class WeightedFusionStrategy(FusionStrategy):
    """
    基于权重的融合策略
    """
    def __init__(self, bm25_weight: float = 0.5, vector_weight: float = 0.5):
        self._bm25_weight = bm25_weight
        self._vector_weight = vector_weight

    def fuse(self, bm25_results: list[RetrievalResult], vector_results: list[RetrievalResult], top_k:int) -> list[RetrievalResult]:
        """
        融合bm25和向量检索结果
        """
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)

        merged_results = self._merged_results(bm25_results, vector_results)
        final_results = []

        for result in merged_results.values():
            fusion_score = result.bm25_score * self._bm25_weight + result.vector_score * self._vector_weight
            result.score = fusion_score
            final_results.append(result) # ?

        sorted_results = sorted(final_results, key=lambda x: x.score, reverse=True)
        # print("fusion result 0 score: ", final_results[0].score)
        # print("fusion result 0 raw_score: ", final_results[0].raw_score)
        # print("fusion result 0 bm25_score: ", final_results[0].bm25_score)
        # print("fusion result 0 vector_score: ", final_results[0].vector_score)
        # print("=" * 80)
        # print("fusion result 0 : \n", final_results[0])
        return sorted_results[:top_k]

    def _normalize_scores(self, results: list[RetrievalResult]) -> list[RetrievalResult]:

        if not results:
            return results

        scores = [result.score for result in results]

        min_score = min(scores)
        max_score = max(scores)

        # 如果所有分数相同，下面的归一化公式就会出现 0/0 的错误
        if min_score == max_score:
            for result in results:
                result.raw_score = result.score
                result.score = 1.0
            return results

        for result in results:
            result.raw_score = result.score
            result.score = (result.score - min_score) / (max_score - min_score)

        return results

    def _merged_results(
        self,
        bm25_results: list[RetrievalResult],
        vector_results: list[RetrievalResult]
    ) -> dict[str, RetrievalResult]:
        merged_results = {}
        for result in vector_results:
            result.vector_score = result.score
            result.bm25_score = 0.0
            merged_results[result.chunk_id] = result
        for result in bm25_results:
            if result.chunk_id in merged_results:
                merged_results[result.chunk_id].bm25_score = result.score
            else:
                result.vector_score = 0.0
                result.bm25_score = result.score
                merged_results[result.chunk_id] = result
        return merged_results