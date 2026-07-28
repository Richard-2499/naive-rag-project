import json
from pathlib import Path
import math

import numpy as np
from llama_index.core.schema import TextNode
from src.logger import get_logger
logger = get_logger(__name__)
"""
vector_store v2建议： 
    state:
        self.index
        self.chunks
        self.config
    
    methods:
        build_index()
        save()
        load()
        get_chunk()
        search() -> [{chunk_id: xxx, score: 0.83}]
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    计算两个向量的余弦相似度
    cosine(θ) = (a · b) / |a| * |b|
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    # np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class VectorStore:
    """
    负责保存和读取向量、向量检索，不负责生成向量

    """
    def __init__(self, save_path):
        '''
        保存向量库文件的位置
        '''
        self.save_path = Path(save_path)
        self.vectors = {} # 为什么要这两行？
        self.chunks = {}

    def save(self, vectors: list[list[float]], nodes: list[TextNode]) -> None:
        assert len(vectors) == len(nodes), "向量数量和文档数量不一致"

        for vector, node in zip(vectors, nodes):
            chunk_id = node.id_
            # save vector
            self.vectors[chunk_id] = vector
            # save chunk info
            self.chunks[chunk_id] = node

        self._save_file()
        logger.info(f"✅ 向量保存完成，保存路径：{self.save_path}")

    def _save_file(self):
        # chunks = [self.chunks[id] = {"text": node.text, "metadata": node.metadata} for id, node in self.chunks.items()]
        # 用新变量chunks, 保持内存状态self.chunks和保存格式chunks分开
        chunks = {}
        for chunk_id, node in self.chunks.items():
            chunks[chunk_id] = {"text": node.text, "metadata": node.metadata}

        data = {
            "vectors": self.vectors,
            "chunks": chunks,
            # next(iter()) 是 Python 中获取字典第一个 value 的标准写法,
            # iter() 把 self.vectors.values()变成迭代器，next() 取第一个
            # 不过实际工程这个字段可以不要，因为 embedding dimension 可以从模型配置拿到，
            # 如 YAML 配置文件中的 embedding: dimension: 512
            # 这里主要是方便debug
            "dimension": len(next(iter(self.vectors.values()))),
            "total": len(self.vectors),
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with self.save_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 向量保存完成，保存路径：{self.save_path}")


    def load(self) -> None:
        from llama_index.core.schema import TextNode
        # reset runtime state
        self.chunks = {}
        self.vectors = {}
        with self.save_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.vectors = data["vectors"]
        # 将 self.chunks 保存成 TextNode
        for chunk_id, chunk in data["chunks"].items():
            self.chunks[chunk_id] = TextNode(
                text = chunk["text"],
                metadata = chunk["metadata"],
                id_ = chunk_id
            )
        logger.info(f"✅ 载入向量完成，向量数量：{data['total']}")

    def get_chunk(self, chunk_id: str):
        return self.chunks[chunk_id]


    def search(self, query_vector: list[float], top_k: int = 3):
        """
        """
        scores: list[tuple[str, float]] = []
        # 2.2 遍历循环每一个document vector
        # 2.3 计算与问题向量的 cosine score
        # 2.4 保存index score
        for chunk_id, vector in self.vectors.items():
            score = _cosine_similarity(query_vector, vector)
            scores.append((chunk_id, score))
        # 2.5 rank
        sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
        # # 2.6 获取top-k scores
        return sorted_scores[:top_k]








