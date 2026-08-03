# 单独新建这个文件，目的是避免
# if model=="bge":
#     xxx
# elif model=="xxx":
#     xxx
# 散落在代码中

from src.reranker.bge_reranker import BGEReranker


class RerankerFactory:

    @staticmethod
    def create(name: str, model_name: str) -> BGEReranker:
        if name == "bge":
            return BGEReranker(model_name = model_name)
        raise ValueError(f"Unsupported reranker: {name}")
