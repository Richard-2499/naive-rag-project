
"""
负责： 计算指标，输出
不负责： 调 Retriever,  读取数据

输入：
    预测结果 + 真实答案
输出： score

Recall@K: 判断是否查全（漏掉多少）
MRR：判断 第一个正确答案在什么位置
Precision@K: 判断 是否查准（准确率）
NDCG@K: 判断 排序质量如何
"""
import math

class Metrics:

    @staticmethod
    def recall_at_k(retrieved_ids:  list[str], relevant_ids: list[str], top_k: int = 5) -> float:
        """
        计算 Recall@K （召回率@K）
        衡量再返回的 K个检索结果中，能覆盖多少真正祥光的文档
        Args:
            retrieved_ids: 检索系统返回的文档 ID 列表 （按相关性从高到底排序）
            relevant_ids: 真正相关的文档 ID 列表（ground truth）
            top_k: 只考虑前 K 个检索结果
        Returns:
            recall@k: 前 K 个结果中相关文档数 / 所有相关文档数
                      范围 [0-1]，值越大表示召回的越全

        Example:
            retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            relevant = ["doc1", "doc3", "doc6"]
            recall_at_k(retrieved, relevant, k=3)
            2/3 = 0.6667  # 前 3个结果中命中了 2 个， doc1 和 doc3， 相关文档数为 3
        """

        # 取前 K 个检索结果（去重）
        retrieved_top_k = set(retrieved_ids[:top_k])
        # 真正的相关文档集合（去重）
        relevant_set = set(relevant_ids)
        if not relevant_set:
            return 0.0
        return len(retrieved_top_k & relevant_set) / len(relevant_set)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        """
        计算 MRR (Mean Reciprocal Rank，平均倒数排名)
        Args:
            retrieved_ids: 检索系统返回的文档 ID 列表（按相关性从高到低排序）
            relevant_ids: 真正相关的文档 ID 列表（ground truth）
        Returns:
            mrr: 第一个相关文档的倒数排名
                 范围 [0, 1]
                 - 1.0: 第一个结果就是相关的（完美）
                 - 0.5: 第二个结果才相关
                 - 0.33: 第三个结果才相关
                 - 0.0: 没有找到任何相关文档
        Note:
            与 Recall@K 和 Precision@K 不同，MRR 只关心第一个正确答案的位置，
            不关心后续结果。适合评估"用户只看第一个结果"的场景。
        """
        relevant_set = set(relevant_ids)
        # 遍历检索结果（按排名顺序）
        for index, chunk_id in enumerate(retrieved_ids, start=1):
            # 找到第一个相关文档
            if chunk_id in relevant_set:
                # 倒数排名 = 1 / 位置索引, 如果索引从0开始，则位置 = index + 1
                return 1 / index
        # 遍历完所有结果都没找到相关文档
        return 0.0

    @staticmethod
    def dcg(relevance_scores: list[int]) -> float:
        score = 0.0
        for index, relevance in enumerate(relevance_scores, start=1):
            score += (2 ** relevance - 1) / math.log2(index+1)
        return score

    @staticmethod
    def ndcg_at_k(reranked_chunk_ids: list[str], ground_truth: dict[str, int], top_k: int) -> float:
        # 获取 dataset 文档中 chunk_id对应的分数
        reranked_scores = [ground_truth.get(chunk_id, 0) for chunk_id in reranked_chunk_ids[:top_k]]
        # 计算 实际排序 DCG
        dcg_score = Metrics.dcg(reranked_scores)

        # 获取理想排序结果
        ideal_scores = sorted(ground_truth.values(), reverse=True)

        # 计算理想配许 IDCG
        idcg_score= Metrics.dcg(ideal_scores[:top_k])

        # 防止没有相关文档导致除零
        if idcg_score == 0:
            return 0.0

        # 返回归一化NDCG
        return dcg_score / idcg_score

