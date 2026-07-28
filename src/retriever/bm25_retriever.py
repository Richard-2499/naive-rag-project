from src.store.bm25_store import BM25Store
from src.schema.retrieval_result import RetrievalResult


class BM25Retriever:
    """
    BM25 检索器，负责用户问题检索，从向量库中与用户问题最相关的top-k 相关文档，不负责回答
    """
    def __init__(self, bm25_store: BM25Store, tokenizer):
        self._bm25_store = bm25_store
        self._tokenizer = tokenizer

    def retrieve(self, query: str, top_k: int = 5):
        # # 1. tokenize query
        # query_tokens = self._tokenizer(query)
        # 2. bm25 search
        scores = self._bm25_store.search(query, top_k)
        # 3. 通过chunk_id 加载文档
        retrieval_results = []
        for chunk_id, score in scores:
            chunk = self._bm25_store.get_chunk(chunk_id)
            result = RetrievalResult(
                chunk_id = chunk_id,
                score = score,
                text = chunk.text,
                metadata = chunk.metadata,
                source = "bm25",
                vector_score = None,
                bm25_score = score
            )
            retrieval_results.append(result)

        # 4. 封装成 RetrievalResult
        return retrieval_results


