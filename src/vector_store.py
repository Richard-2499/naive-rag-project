import json
from pathlib import Path

from llama_index.core.schema import TextNode
from sklearn.metrics.pairwise import cosine_similarity
from src.logger import get_logger
logger = get_logger(__name__)
class VectorStore:
    """
    负责保存和读取向量、向量检索，不负责生成向量
    """
    def __init__(self, save_path):
        '''
        保存向量库文件的位置
        '''
        self.save_path = Path(save_path)  # 与 = save_path 区别？

    def save(self, vectors: list[list[float]], nodes: list[TextNode]) -> None:
        assert len(vectors) == len(nodes), "向量数量和文档数量不一致"
        nodes_data = [
            {"text": node.text,
             "metadata": node.metadata,
             "id": node.id_,
             } for node in nodes
        ]
        data = {
            "vectors": vectors,
            "nodes": nodes_data,
            "dimension": len(vectors[0]),
            "total": len(vectors),
        }
        with self.save_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 向量保存完成，保存路径：{self.save_path}")

    def load(self) -> dict:
        with self.save_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"✅ 载入向量完成，向量数量：{data['total']}")
        return data

    def search(self, query_vector: list[float], top_k: int = 3) -> list[TextNode]:
        """
        将 query_vector 和 文档的向量 做相似度检索，返回 top_k 个相关文档
        Args:
            query_vector: 用户问题的向量
            top_k:  相关文档的个数
        Returns:
             top-k 相关文档 list[Document]
        步骤：
        1. 先拿当前vectorStore 中的内容
        2. 用户问题向量与所有向量做相似度检查（cosine/L2)
            2.1 读取vector,
            2.2 遍历循环每一个document vector
            2.3 计算与问题向量的 cosine score
            2.4 保存index score
            2.5 rank
            2.6 取top-k
        3. 再把top-k 个原文向量对应文档块获取到
        4. 返回top-k 个文档
        """
        data = self.load()
        # print(type(data)) # <class 'dict'>
        # print(data.keys()) # dict_keys(['vectors', 'nodes', 'dimension', 'total'])
        all_vectors = data["vectors"]
        nodes = data["nodes"]
        assert len(all_vectors) == len(nodes), "向量数量和文档数量不一致"
        # print(texts[29][:300])  # 节点30
        scores: list[tuple[int, float]] = []
        # 2.2 遍历循环每一个document vector
        # 2.3 计算与问题向量的 cosine score
        # 2.4 保存index score
        for i, vector in enumerate(all_vectors):
            score = cosine_similarity([query_vector], [vector])[0][0]
            scores.append((i, score))
        # 2.5 rank
        sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)
        # 2.6 获取top-k ids
        top_k_ids = [x[0] for x in sorted_scores[:top_k]]
        # 3 取top-k 对应node
        top_k_nodes = [nodes[i] for i in top_k_ids]

        # 4 返回top-k 文档
        return [TextNode(text = item["text"], metadata = item["metadata"], id_ = item["id"]) for item in top_k_nodes]
        # return top_k_nodes
        # result = []
        # for i,j,k in zip(top_k_ids, top_k_texts, sorted_scores[:top_k]):
        #     result.append([(i,k[1], j)])
        # return result if result else top_k_texts







