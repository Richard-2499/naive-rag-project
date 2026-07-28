# import json
# from pathlib import Path
#
# from llama_index.core.schema import TextNode
# from src.logger import get_logger
# logger = get_logger(__name__)
#
#
# """
# 保存文本索引结构
# 建立倒排索引 BM25 index
# keyword search
#
# """
#
# class BM25Store:
#     """
#     Responsibilities:
#         - store chunks
#         - build inverted index
#         - BM25 search
#         - persistence
#
#     Args:
#         save_path:
#             json storage path
#
#         tokenizer:
#             documents/query tokenizer
#
#     Runtime:
#         chunks:
#             chunk_id -> TextNode
#
#         inverted_index:
#             term -> {chunk_id: frequency}
#         Example:
#             {
#                 "模型"：{
#                     "chunk_001": 5,
#                     "chunk_002": 7
#                 }
#             }
#
#         doc_lengths:
#             chunk_id -> token 数量
#
#     """
#     def __init__(self, save_path: str, tokenizer):
#         self._save_path = Path(save_path)
#         self._tokenizer = tokenizer
#         self.chunks = {}
#         self.chunk_ids = []
#         self.corpus = [] # 这是啥？
#         self.bm25 = None # bm25 引擎
#
#     # Public API
#     def save(self, nodes: list[TextNode]):
#         """
#         index 持久化
#         不能 在启动服务时， load PDF -> 重新 tokenize -> 重新 bulid BM25 -> 等待启动， 而是：
#         启动 -> load bm25 index -> ready
#
#         Returns:
#             BM25 search results
#         """
#         self.chunks = {}
#         self.chunk_ids = []
#         self.corpus = []
#
#         for node in nodes:
#             # 保存chunk
#             chunk_id = node.id_
#             self.chunks[chunk_id] = node
#             self.chunk_ids.append(chunk_id)
#             # tokenize
#             tokens = self._tokenize(node.text)
#             self.corpus.append(tokens)
#
#         # build BM25 index
#         self.bm25 = BM25Okapi(self.corpus)
#         self._save_file()
#
#     def load(self):
#         """
#         负责：
#             恢复：
#                 tokenizer config
#                 inverted index
#                 document frequency
#                 chunk metadata
#         """
#         with self._save_path.open("r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         # reset runtime state
#         self.chunks = {}
#         # retore chunks
#         for chunk_id, node in data["chunks"].items():
#             self.chunks[chunk_id] = TextNode(
#                 text = node["text"],
#                 metadata = node["metadata"],
#                 id_ = chunk_id
#             )
#         self.inverted_index = data["inverted_index"]
#         self.doc_lengths = data["doc_lengths"]
#         self.avg_doc_length = data["avg_doc_length"]
#
#     def search(self, query, top_k: int = 5) -> list[tuple[str, float]]:
#         """
#         BM25 search
#         """
#         query_tokens = self._tokenize(query)
#         # 根据inverted index 找到candidate chunk
#         candidate_chunks = set()
#         for token in query_tokens:
#             if token in self.inverted_index:
#                 candidate_chunks.update(self.inverted_index[token].keys())
#         # 计算 BM25 score
#         scores = {}
#         for chunk_id in candidate_chunks:
#             # 获取 chunk length
#             doc_length = self.doc_lengths[chunk_id]
#             # 获取 chunk frequency
#             chunk_freq = sum(self.inverted_index[token][chunk_id] for token in query_tokens if token in self.inverted_index)
#             # 计算 BM25 score
#             score = chunk_freq / (chunk_freq + 0.5 + (1.5 / self.avg_doc_length))
#             # 保存 score
#             scores[chunk_id] = score
#
#         # 排序
#         ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#
#         return ranked[:top_k]
#
#     def get_chunk(self, chunk_id: str):
#         """
#         获取 chunk
#         """
#         return self.chunks[chunk_id]
#
#     # Private API
#     def _tokenize(self, text: str) -> list[str]:
#         """
#         tokenizer wrapper
#         后续如果换:
#         jieba
#         nltk
#         transformers tokenizer
#         只改这里
#         """
#
#
#         result = self._tokenizer.tokenize(text)
#         return result
#     def _save_file(self):
#         """
#         保存 index
#         """
#         chunks = {}
#         for chunk_id, node in self.chunks.items():
#             chunks[chunk_id] = {"text": node.text, "metadata": node.metadata}
#         data = {
#             "chunks": chunks,
#             "inverted_index": self.inverted_index,
#             "doc_lengths": self.doc_lengths,
#             "avg_doc_length": self.avg_doc_length
#         }
#         self._save_path.parent.mkdir(parents=True, exist_ok=True)
#         with self._save_path.open("w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=2)
#


