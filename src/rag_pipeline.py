import logging

from config.config_loader import load_config
from src.generation import Generation
from src.reranker.base_reranker import BaseReranker
from src.retriever.hybrid_retriever import HybridRetriever
from src.schema.retrieval_result import RetrievalResult

config = load_config()
from src.logger import get_logger, get_log_level
level = get_log_level(config["logging"]["level"])
logger = get_logger(__name__, level)
# logger = get_logger(__name__, logging.INFO)

class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generation: Generation, reranker: BaseReranker) -> None:
        self._retriever = retriever
        self._generation = generation
        self._reranker = reranker

    def query(self, query: str) -> str:
        candidates = self._retriever.retrieve(query, top_k=config["retrieval"]["hybrid"]["candidate_k"])
        logger.debug(
            "Hybrid retrieval:\n%s",
            self.format_retrieval_results(candidates)
        )
        reranked_results = self._reranker.rerank(query = query, results = candidates, top_k = config["retrieval"]["hybrid"]["top_k"])
        logger.debug(
            "Reranker results: \n%s",
            self.format_retrieval_results(reranked_results)
        )
        answer = self._generation.generate(query, reranked_results)
        return answer
    @staticmethod
    def format_retrieval_results(results: list[RetrievalResult]):
        lines = []
        for i, item in enumerate(results, start=1):
            lines.append(
                f"{i},"
                f"id = {item.chunk_id},"
                f"score = {item.score:.4f},"
                f"source = {item.source}"
            )
        return "\n".join(lines)