from llama_index.core.schema import TextNode

from src.loader import *
from config.config_loader import load_config
config = load_config()

from src.bge_embedder import BGEEmbedder
from src.splitter import Splitter
from src.store.vector_store import VectorStore
# file_path = config["paths"]["raw_data"]
# docs = DocumentLoader(file_path).load_documents()
embedder = BGEEmbedder(config["embedding"]["model"], nomalize=True)
# nodes = Splitter(config["splitter"]["chunk_size"], config["splitter"]["chunk_overlap"]).split_documents(docs)
nodes = Splitter.load_nodes_from_file(config["paths"]["chunks_store"])

texts = [node.text for node in nodes]
embedding_data = embedder.embed(texts)
new_vector_store = VectorStore(config["paths"]["vector_store"])
new_vector_store.save(embedding_data, nodes)
# vector_data = new_vector_store.load()

# print(f"向量数量：{len(vector_data["vectors"])}") # 106
# print(f"类型：{type(vector_data["vectors"][10])}") # <class 'list'>
# print(f"向量维度：{len(vector_data["vectors"][20])}") # 512
# print(f"文本验证：{vector_data['texts'][22][:300]}") # 文本验证：规范性文档： - "应该怎么做"    - 示例：标注规范、质量标准   描述性文档：
# query = "JIT的三个层次"
# query_embedding = embedder.embed([query])
# # print(f"query embedding type is : {type(query_embedding)}")
#
# top_k_texts = new_vector_store.search(query_embedding[0], top_k=20)
# # print(f"返回结果类型：{type(top_k_texts)}") # <class 'list'>
# # print(f"index 2 type: {type(top_k_texts[0])}") # index 2 type: <class 'llama_index.core.schema.TextNode'>
# assert isinstance(top_k_texts, list), "top_k_texts is not a list"
# assert isinstance(top_k_texts[0], TextNode), "top_k_texts is not a Document"
#
# print("=" * 60)
# print(f"user query: {query}")
# print("=" * 60)
# for i, item in enumerate(top_k_texts[:5]):
#     print(f"第{i+1}个最相似文本：\n{item.text}")
#     print("=" * 60)
# # for i in top_k_texts:
# #     print(i[0][2])