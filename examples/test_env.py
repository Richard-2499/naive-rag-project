'''
test_env.py --- 测试 API key 和 模型

'''
# from src.draft.config import Config

# print(key[:30])
# client = OpenAI(
#     api_key=key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )
#
# response = client.chat.completions.create(
#     model=Config.LLM_MODEL,
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the meaning of life?"},
#     ],
#     temperature=0.6,
#     max_tokens=1024,
# )
# print(response.choices[0].message.content)

#
# response = client.chat.completions.create(
#     model="gemini-2.5-flash",
#     messages=[
#         {
#             "role":"user",
#             "content":"Hello"
#         }
#     ]
# )

# jina_key = os.getenv("JINA_API_KEY")
# jina_embed_url = "https://api.jina.ai/v1"
# jina_llm_url = "https://deepsearch.jina.ai/v1"
# # 联网大模型
# llm_client = OpenAI(
#     api_key=jina_key,
#     base_url=jina_llm_url,
# )
# # 向量嵌入模型
# embed_client = OpenAI(
#     api_key=jina_key,
#     base_url=jina_embed_url,
# )
#
# # 用户问题向量化
# texts = ["我爱你"]
#
# # 2. 调用向量化接口（OpenAI 兼容格式）
# embeddings = embed_client.embeddings.create(
#     model = "jina-embeddings-v4",
#     input = texts,
#     extra_body = {
#         "task": "retrieval.passage",  # 文档入库用
#         "truncate_dim": 1024
#     }
# )
#
# # 3. 解析返回结果（和OpenAI返回结构完全一致）
# print("消耗Token：", embeddings.usage.total_tokens)
#
# for item in embeddings.data:
#     print("向量维度：", len(item.embedding))
#     print("向量值：", item.embedding)
# user_query = "Jina向量和大模型接口域名分别是什么，免费额度规则？清晰完整回答"
#
#
# llm_resp = llm_client.chat.completions.create(
#     model=Config.JINA_LLM_MODEL,
#     messages=[
#         {
#             "role":"user",
#             "content": "你是谁？"
#         }
#     ],
#     temperature=0.6,
# )
# print("消耗Token：", llm_resp.usage.total_tokens)
# print("====== LLM 模型回答 ======")
# print(llm_resp.choices[0].message.content)

# from sentence_transformers import SentenceTransformer
#
# model = SentenceTransformer('BAAI/bge-small-zh')
# vector = model.encode("测试文本")
#
# print(f"向量维度: {len(vector)}")
# print(f"前5个值: {vector[:5]}")


# vectors = {"a": "aaa", "b": "bbb", "c": "ccc"}
# # for i in vectors:
# #     # print(i)
# #     print(vectors[i])
# bb = [vectors[i] for i in vectors]
# print(bb)

# vector = {"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6], "c": [0.7, 0.8, 0.9]}
# dim1 = len(vector)
# dim2 = len(next(iter(vector.values())))
# print(dim1, dim2)

scores = [0.22, 0.33, 0.44, 0.62, 0.93, 0.81, 0.12, 0.59, 0.73, 0.06]

sorted_scores = sorted(scores, key=lambda x: x, reverse=True)
print(sorted_scores)
sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
print(sorted_indices)