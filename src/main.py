# src/main.py
import os
from pathlib import Path
from bge_embedder import BGEEmbedder
from config.config_loader import load_config
from vector_store import VectorStore
from retriever import Retriever
from generation import Generation, LLMClient
from rag_pipeline import RAGPipeline


def main():
    config = load_config()
    # 1. 初始化 Embedding
    embedder = BGEEmbedder("BAAI/bge-small-zh", nomalize=True)

    # 2. 初始化 VectorStore
    vector_store = VectorStore(
        save_path= config["paths"]["vector_store"]
    )

    # 3. 初始化 Retriever
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # 4. 初始化 LLM
    llm = LLMClient(
        model="deepseek-v4-flash",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=config["llm"]["base_url"],
    )

    # 5. 初始化 Generation
    generation = Generation(
        llm=llm
    )

    # 6. 组装 RAG Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        generation=generation,
    )

    # 7. 用户问题
    question = "JIT 的三个层次？"

    answer = pipeline.query(question)

    print("\n回答:")
    print(answer)


if __name__ == "__main__":
    main()