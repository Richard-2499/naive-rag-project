import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
from llama_index.core.schema import TextNode

class BM25Store:
    def __init__(self, save_path: str, tokenizer):
        """
        BM25 Store

        负责:
        - store document chunks
        - build BM25 index
        - execute BM25 search
        - persistence

        不负责:
        - embedding
        - fusion
        - reranking
        """
        self.save_path = Path(save_path)
        self.tokenizer = tokenizer # same tokenizer for: document indexing, query searching

        # Runtime Data

        # chunk_id -> TextNode
        self.chunks: dict[str, TextNode] = {}
        # BM25 returns score by index, maintain mapping: index -> chunk_id
        self.chunk_ids: list[str] = []

        # tokenized documents: eg, [["机器学习", "模型"], ["深度学习", "模型"]...]
        self.corpus: list[list[str]] = []
        self.bm25 = None # bm25 引擎，created after build index

    def save(self, nodes: list[TextNode]) -> None:
        """
        Build BM25 index and save.
        Flow:
            TextNode
                ↓
            tokenize(text)
                ↓
            corpus
                ↓
            BM25Okapi
        """
        # reset runtime state
        self.chunks = {}
        self.chunk_ids = []
        self.corpus = [] # bm25 的输入语料库（token后的文档集合），作为入参传给BM25Okapi

        for node in nodes:
            chunk_id = node.id_
            # store original chunk
            self.chunks[chunk_id] = node
            # keep index mapping
            self.chunk_ids.append(chunk_id)
            # tokenize document
            tokens = self._tokenize(node.text)
            self.corpus.append(tokens)

        # build BM25 engine
        self.bm25 = BM25Okapi(self.corpus)
        self._save_file()

    def load(self) -> None:
        """
        Restore BM25 Store.
        注意：load完成后要转化为TextNode类型，与保存前状态保持一致
        """
        with self.save_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # reset and restore chunks
        self.chunks = {}
        for chunk_id, chunk_data in data["chunks"].items():
            self.chunks[chunk_id] = TextNode(
                text=chunk_data["text"],
                metadata=chunk_data["metadata"],
                id_=chunk_id
            )
        self.chunk_ids = data["chunk_ids"]
        self.corpus = data["corpus"]
        # restore BM25 engine
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        BM25 search.
        Return:
            [(chunk_id, score)]
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not initialized")
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens) # list[float]

        # 对scores排序并返回对应的index
        ranked_indices = sorted(range(len(scores)), key = lambda x: scores[x], reverse = True)
        results: list[tuple[str, float]] = []

        # 返回前top_k个结果 [(chunk_id, score)]
        for idx in ranked_indices[:top_k]:
            results.append(
                (self.chunk_ids[idx], float(scores[idx]))
            )
        return results

    def get_chunk(self, chunk_id: str) -> TextNode:
        return self.chunks[chunk_id]

    def _tokenize(self, text: str) -> list[str]:
        return self.tokenizer(text)

    def _save_file(self) -> None:
        """
        Persistence.
        Runtime:
            TextNode

        Convert:
            serializable dict
        """
        chunks = {}
        for chunk_id, node in self.chunks.items():
            chunks[chunk_id] = {
                "text": node.text,
                "metadata": node.metadata
            }
        data = {
            "chunks": chunks,
            "chunk_ids": self.chunk_ids,
            "corpus": self.corpus
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with self.save_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)