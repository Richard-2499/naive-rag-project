import os
os.environ['HF_HUB_OFFLINE'] = '1'  # Hugging Face 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Transformers 离线模式
from sentence_transformers import SentenceTransformer

class BGEEmbedder:
    """
    基于 BGE 模型进行向量化，不负责存储
    """
    def __init__(self, model_path, nomalize: bool = True):
        """
        初始化 Embedding 模型
        """
        self.model = SentenceTransformer(model_path)
        self.nomalize = nomalize

    def embed(self, texts: list[str] | str) -> list[list[float]]:
        '''
        将文本转换为向量
        '''
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        return self.model.encode(texts, normalize_embeddings=self.nomalize).tolist()

