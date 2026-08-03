import os
os.environ['HF_HUB_OFFLINE'] = '1'          # Hugging Face 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'    # Transformers 离线模式
from sentence_transformers import CrossEncoder

from config.config_loader import load_config
from src.reranker.base_reranker import BaseReranker
from src.schema.retrieval_result import RetrievalResult

config = load_config()

class BGEReranker(BaseReranker):
    """
    BGE Cross Encoder Reranker

    输入： query + results
    输出： relevance score
    """
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
        self.model: CrossEncoder = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results or top_k <= 0:
            return []
        pairs: list[list[str]] = [
            [query, item.text] for item in results
        ]
        scores = self.model.predict(pairs)
        for item, score in zip(results, scores):
            item.rerank_score = float(score)
            item.rerank_source = "reranker"
            item.is_reranked = True

        reranked_results: list[RetrievalResult] = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        return reranked_results[:top_k]

