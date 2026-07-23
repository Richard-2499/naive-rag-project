'''
embedder.py  - 向量化器
将切分好的文本块转换成向量，并构建索引
'''
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.dashscope import DashScopeEmbedding
from src.draft.config import Config

def set_models():
    Settings.embed_model = DashScopeEmbedding(
        api_key=Config.JINA_API_KEY,
        model_name = Config.JINA_EMBEDDING_MODEL
    )

    # Settings.llm_model = DashScope(
    #     api_key=Config.JINA_API_KEY,
    #     model_name = Config.JINA_LLM_MODEL,
    #     temperature = Config.TEMPERATURE,
    #     max_tokens = Config.MAX_TOKENS
    # )


def build_index(nodes):
    '''
    从切分好的节点构建向量索引

    Args： splitter.py 返回的节点列表

    Returns：
        VectorStoreIndex 对象
    '''
    # 1. 检查 nodes 是否为空，如果为空 raise ValueError
    if not nodes:
        raise ValueError("nodes 为空，请检查splitter.py的返回结果")
    # 2. 打印 "正在构建向量索引..."
    print("正在构建向量索引...")
    # 3. 用 VectorStoreIndex(nodes=nodes, show_progress=True) 构建索引
    embeddings = VectorStoreIndex(nodes, show_progress=True)
    # 4. 打印索引构建完成信息（共 {len(index)} 个向量）
    print(f"索引构建完成，共 {len(embeddings)} 个向量")
    # 5. 返回 index
    return embeddings

def build_index_from_file(file_path):
    '''
    从文件中加载节点，并构建向量索引

    Args：
        file_path: 文件路径

    Returns：
        VectorStoreIndex 对象
    '''
    # 1. 检查 file_path 是否为空，如果为空 raise ValueError
    if not file_path:
        raise ValueError("file_path 为空，请检查参数")
    # 2. 用 VectorStoreIndex.from_documents(documents, show_progress=True) 构建
    embeddings = VectorStoreIndex.from_documents(file_path, show_progress=True)
    # 3. 返回 index
    return embeddings

if __name__ == "__main__":
    # 1. 从 loader 导入 load_documents
    from loader import load_documents
    # 2. 从 splitter 导入 split_documents
    from splitter import split_documents, load_validation_questions

    # 3. 打印测试标题
    print("测试：向量化器")
    # 4. 加载文档 → docs = load_documents()
    loads = load_documents()
    # 5. 切分文档 → nodes = split_documents(docs)
    nodes = split_documents(loads)
    # 6. 配置模型 → setup_models()
    set_models()
    # 7. 构建索引 → index = build_index(nodes)
    embeddings = build_index(nodes)
    # 8. 打印测试通过
    print("测试通过")