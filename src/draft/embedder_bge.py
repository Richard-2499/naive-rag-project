import os
import time
from src.draft.config import Config
os.environ['HF_HUB_OFFLINE'] = '1'          # Hugging Face 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'    # Transformers 离线模式
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. 加载模型
# ============================================================
# BGE 模型选项：
#   - "BAAI/bge-small-zh"   : 轻量，512维，速度快（推荐）
#   - "BAAI/bge-base-zh"    : 中等，768维
#   - "BAAI/bge-large-zh"   : 最大，1024维，效果最好
embed_model = SentenceTransformer('BAAI/bge-small-zh')

# 2. 文本向量化
def get_embedding(text: str, model: str = embed_model) -> list[float]:
    """
    将单条文本转为向量（BGE 本地推理）

    Args:
        text: 要向量化的文本
        model_name: BGE 模型名称

    Returns:
        向量列表 (list[float])
    """
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# 3. 批量向量化
def get_embeddings_batch(
        texts: list[str],
        model: str = embed_model,
        batch_size: int = 20,
        on_batch_complete = None,
) -> list[list[float]]:
    """
    批量将多条文本转为向量（BGE 本地推理）

    Args:
        texts: 文本列表
        model_name: BGE 模型名称
        batch_size: 每批处理的数量

    Returns:
        向量列表，每个元素是一个向量
    """
    embeddings = []
    total = len(texts)
    total_tokens = 0
    tokenizer = model.tokenizer
    print(f"🔨 开始向量化 {total} 条文本（BGE 本地模型）...")
    for i in range(0, total, batch_size):
        text_batch = texts[i:i+batch_size]
        # 统计当前批次的token 消耗
        batch_tokens = sum(len(tokenizer.encode(t)) for t in text_batch)
        total_tokens += batch_tokens
        embedding = model.encode(text_batch, normalize_embeddings=True)  # 返回向量列表， list[list[float]] 格式
        # embeddings.extend(embedding.tolist())  # 不加tolist，则返回的是numpy, 存json文件时，会出错

        done = min(i+batch_size, len(texts))
        time.sleep(0.1)
        if on_batch_complete:
            on_batch_complete(embedding.tolist(), done)
        else:
            embeddings.extend(embedding.tolist())
        print(f" 进度： 已处理 {done}/{len(texts)} 条数据， 本批 token消耗： {batch_tokens}, 累计： {total_tokens}")
    print(f"✅ 总 token 数: {total_tokens:,}, 平均每条: {total_tokens // total}")
    return embeddings if not on_batch_complete else None

# 4. 构建向量存储
def build_vector_store(
        nodes,
        model: str = embed_model,
        checkpoint_file: str = "vector_store_checkpoint.json",
        cache_file: str = "vector_store_cache.json",
        batch_size: int = 20,
) -> dict:
    """
    从切分好的 nodes 构建向量存储
    支持：
      1. 缓存加载：如果已有完整结果，直接加载，跳过计算
      2. 断点续传：如果中途中断，从上次保存的位置继续
    Args:
        nodes: splitter.py 返回的节点列表
        model: BGE 模型
        checkpoint_file: 断点续传文件路径
        cache_file: 保存的json文件
        batch_size: 每批处理的数量

    Returns:
        向量存储字典
    """

    import json
    import os
    # ============================================================
    # 第一步：检查是否有完整的缓存文件
    # ============================================================
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            # 验证缓存数据是否完整
            if "vectors" in cached_data and "texts" in cached_data and len(cached_data["vectors"]) == len(nodes) and len(nodes) > 0:
                print(f"✅ 发现完整缓存文件 {cache_file}，直接加载（跳过向量化）")
                print(f"   - 向量数量: {len(cached_data['vectors'])}")
                print(f"   - 向量维度: {cached_data.get('dimension', '未知')}")
                return cached_data
        except Exception as e:
            print(f"⚠️ 缓存文件读取失败: {e}，重新向量化")
    # ============================================================
    # 第二步：检查断点文件（支持续传）
    # ============================================================
    texts = [node.text for node in nodes]
    total = len(texts)

    # ===== 加载已有进度 =====
    start_idx = 0
    all_vectors = []

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            all_vectors = checkpoint.get("vectors", [])
            start_idx = len(all_vectors)
            print(f"📂 发现断点文件，已向量化 {start_idx}/{total} 条，从断点继续...")
        except Exception as e:
            print(f"⚠️ 断点文件损坏，从头开始（错误: {e}）")
            start_idx = 0
            all_vectors = []

    # 如果已经全部完成，直接返回
    if start_idx >= total:
        print(f"✅ 向量化已完成！共 {total} 条")
        dimension = len(all_vectors[0]) if all_vectors else 0
        result = {
            "vectors": all_vectors,
            "texts": texts,
            "dimension": dimension,
            "total": len(all_vectors),
        }
        # with open(cache_file, "w", encoding="utf-8") as f:
        #     json.dump(result, f, ensure_ascii=False)
        return result

    # ============================================================
    # 第三步： 从断点继续向量化
    # ============================================================
    print(f"🔨 开始向量化 {total} 条文本（从第 {start_idx + 1} 条开始）...")
    def save_checkpoint(embedding: list[list[float]], done):
        nonlocal all_vectors
        all_vectors.extend(embedding)
        # 每批完成后立即保存进度
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"vectors": all_vectors}, f, ensure_ascii=False)

    get_embeddings_batch(texts[start_idx:], model, batch_size=batch_size, on_batch_complete=save_checkpoint)
    print(f"  💾 断点已保存， 共 {len(all_vectors)} 条")

    # ============================================================
    # 第四步：全部完成，保存缓存并删除断点文件
    # ============================================================
    # 返回完整结果
    dimension = len(all_vectors[0]) if all_vectors else 0
    result=  {
        "vectors": all_vectors,
        "texts": texts,
        "dimension": dimension,
        "total": len(all_vectors),
    }
    # 保存缓存文件
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # 删除断点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✅ 全部完成，断点文件已清理")
    print(f"✅ 全部完成！共 {total} 条向量，已保存到 {cache_file}")
    return result

if __name__ == "__main__":
    from loader import load_documents
    from splitter import split_documents
    USE_SAMPLE = 1
    if USE_SAMPLE:
        SAMPLE_SIZE = 100
    print("=" * 60)
    print("测试：手搓版向量化器")
    print("=" * 60)

    documents = load_documents()
    nodes = split_documents(documents)
    nodes = nodes[:SAMPLE_SIZE] if USE_SAMPLE else nodes
    print(f"实际需要向量化的节点数量： {len(nodes)}")

    result = build_vector_store(nodes)

    print(f"向量维度： {result["dimension"]}")
    print(f"向量数量： {result["total"]}")
    if result["total"] > 0:
        print(f"第一个向量前5个值： {result["vectors"][0][:5]}")
        print(f"第一个文本前50个字： {result["texts"][0][:50]}")
    print("消耗Token：", result["total"] * result["dimension"] * 4 / 1000)
    print("\n ✔ 测试通过")

