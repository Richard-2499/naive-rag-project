from src.generation import Generation
from src.retriever.hybrid_retriever import HybridRetriever


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, generation: Generation, top_k = 20) -> None:
        self._retriever = retriever
        self._generation = generation
        self._top_k = top_k

    def query(self, question: str) -> str:
        nodes = self._retriever.retrieve(question, top_k=self._top_k)
        answer = self._generation.generate(question, nodes)
        return answer
