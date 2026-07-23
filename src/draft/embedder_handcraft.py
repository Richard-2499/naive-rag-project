import time
from openai import OpenAI
from src.draft.config import Config

# 1. 创建客户端

def create_llm_client():
    client = OpenAI(
        api_key=Config.JINA_API_KEY,
        base_url=Config.JINA_LLM_URL,
    )
    return client

def create_embed_client():
    client = OpenAI(
        api_key=Config.DASHSCOPE_API_KEY,
        base_url=Config.QWEN_BASE_URL,
    )
    return client

embed_client = create_embed_client()

# 2. 文本向量化
def get_embedding(client: OpenAI, text: str, model: str = Config.QWEN_EMBEDDING_MODEL) -> list[float]:
    # 单条文本向量化
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


# 3. 批量向量化
def get_embeddings_batch(client: OpenAI, texts: list[str], model: str = Config.QWEN_EMBEDDING_MODEL, batch_size: int = 20) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        text_batch = texts[i:i+batch_size]
        embedding = client.embeddings.create(input=text_batch, model=model)  # 返回向量列表， list[list[float]] 格式
        for item in embedding.data:
            embeddings.append(item.embedding)

        done = min(i+batch_size, len(texts))
        print(f"已处理 {done}/{len(texts)} 条数据")
        time.sleep(0.1)
    return embeddings



# 4. 构建向量存储
# def build_vector_store(client: OpenAI, texts: list[str], model: str = Config.JINA_EMBEDDING_MODEL):
#     pass

def build_vector_store(nodes) -> dict:
    texts = [node.text for node in nodes]
    client = create_embed_client()
    embeddings = get_embeddings_batch(client, texts)

    result = {
        "vectors": embeddings,
        "texts": texts,
        "dimension": len(embeddings[0]) if embeddings else 0,
        "total": len(embeddings),
    }
    return result

if __name__ == "__main__":
    from loader import load_documents
    from splitter import split_documents
    USE_SAMPLE = True
    if USE_SAMPLE:
        SAMPLE_SIZE = 100
    print("=" * 60)
    print("测试：手搓版向量化器")
    print("=" * 60)

    documents = load_documents()
    nodes = split_documents(documents)
    result = build_vector_store(nodes[:SAMPLE_SIZE])

    # save to file
    import json
    with open(f"vectors_store_{SAMPLE_SIZE}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"向量存储已保存到 vectors_store_{SAMPLE_SIZE}.json")



    print(f"向量维度： {result["dimension"]}")
    print(f"向量数量： {result["total"]}")
    if result["total"] > 0:
        print(f"第一个向量前5个值： {result["vectors"][0][:5]}")
        print(f"第一个文本前50个字： {result["texts"][0][:50]}")
    print("消耗Token：", result["total"] * result["dimension"] * 4 / 1000)
    print("\n ✔ 测试通过")



