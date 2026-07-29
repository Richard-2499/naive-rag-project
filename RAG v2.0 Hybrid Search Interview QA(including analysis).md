**这个 RAG 项目本身不是特别复杂**。

但这里有一个关键点：

> 面试不是考你“做了多复杂的 RAG”，而是考你有没有理解 RAG 工程链路、为什么这么设计、遇到问题怎么解决。

你的项目定位：

​	**不是“我做了一个超级复杂的 AI 系统”**
而是：

> “我从 0 到 1 手写实现了一个模块化 Hybrid Search RAG，并通过这个项目理解企业 RAG 的核心工程设计。”

这样定位是合理的。

下面是整理版本：

# RAG Hybrid Search 项目面试题库（含答案+回答思路）

------

# Q1：请介绍一下你做的 RAG 项目？

## 推荐回答（2分钟）

> 我实现了一个基于 Hybrid Search 的 RAG 系统，主要解决企业知识库问答场景下，大模型无法直接访问私有知识的问题。
>
> 整体流程包括文档加载、文本清洗、chunk切分、embedding生成、向量检索、BM25检索、结果融合以及LLM生成。
>
> 第一版实现的是传统 Vector RAG，通过 embedding 做语义检索，但是测试过程中发现对于专业术语、产品型号、缩写等精确匹配场景召回效果不足。
>
> 所以后续升级为 Hybrid Search，将 Vector Search 和 BM25 Keyword Search 结合，通过 Fusion Strategy 对两个检索结果进行归一化和加权融合，提高召回效果。
>
> 在工程设计上，我把 Retriever、Store、Fusion、Pipeline 进行解耦，通过统一的 RetrievalResult 数据结构传递检索结果，提高系统扩展性。
>
> 目前支持离线构建索引，在线查询，并且可以方便扩展 reranker、evaluation 等模块。

------

## 回答思路

不要按照代码讲：

错误：

> 我写了loader.py，然后vector_store.py，然后bm25.py...

面试官不关心文件。

按照：

```
为什么做
 ↓
遇到什么问题
 ↓
怎么优化
 ↓
架构怎么设计
 ↓
有什么扩展
```

------

# Q2：为什么从 Vector RAG 升级到 Hybrid Search？

## 答案

> 单纯 Vector Search 主要依赖 embedding 的语义相似度，对于语义理解效果比较好，但是对于一些精确关键词匹配场景，比如产品型号、错误码、API名称、专业缩写，召回能力可能不足。
>
> BM25 基于词频和逆文档频率，对于关键词匹配更加敏感。
>
> 所以 Hybrid Search 将两者结合：
>
> Vector Search 负责语义召回；
>
> BM25负责关键词召回；
>
> Fusion负责综合排序。

------

## 思路

关键词：

```
Vector:
理解意思

BM25:
匹配文字
```

不要说：

> BM25更准确

错误。

应该说：

> 两者适合不同场景。

------

# Q3：你的 RAG 整体架构是什么？

## 答案

```
Document

 ↓

Loader

 ↓

Cleaning

 ↓

Chunking

 ↓

Embedding

 ↓

Index


====================


Query

 ↓

Pipeline

 ↓

HybridRetriever

 ↓

 ----------------
 VectorRetriever

 BM25Retriever
 ----------------

 ↓

Fusion

 ↓

Top-K Context

 ↓

Prompt

 ↓

LLM
```

------

## 设计解释：

核心模块：

### Store

负责数据存储：

- vector
- bm25 index

### Retriever

负责搜索：

- query处理
- 调用store

### Fusion

负责排序：

- normalize
- merge

### Pipeline

负责流程编排。

------

# Q4：为什么 Retriever 和 Store 要分开？

## 答案

> Store主要负责数据存储和底层查询能力，而Retriever负责业务层检索逻辑。
>
> 分离以后，可以替换底层存储而不影响上层逻辑，例如VectorStore可以替换成Milvus、FAISS，而Retriever接口保持不变。

------

## 面试考察：

架构抽象能力。

------

# Q5：为什么设计 RetrievalResult？

## 答案

> 因为 Hybrid Search 有多个检索来源，如果每个Retriever返回不同格式，Fusion会产生大量兼容代码。
>
> 所以设计统一的数据结构，包含chunk_id、text、metadata、score以及不同来源score，使Vector、BM25和后续Reranker可以统一处理。

------

# Q6：为什么不用VectorRetriever直接返回文本？

## 答案

> 因为文本只是最终展示内容，检索系统还需要保留更多信息，比如chunk id用于去重，metadata用于过滤，score用于排序和debug。
>
> 所以返回结构化结果更加适合工程系统。

------

# Q7：Fusion为什么需要Normalization？

## 答案

> 因为不同检索算法的score没有统一尺度。
>
> Vector Search通常是cosine similarity，例如0到1。
>
> BM25 score可能是几到几十。
>
> 如果直接相加，会导致BM25占主导。
>
> 所以需要先归一化，再进行weighted fusion。

------

# Q8：Hybrid Search如何确定权重？

例如：

```python
score =
0.7 * vector_score +
0.3 * bm25_score
```

## 答案

> 权重通常需要根据业务数据调优。
>
> 如果知识库专业术语较多，可以提高BM25权重。
>
> 如果用户问题表达比较自然，可以提高Vector权重。
>
> 实际生产中一般通过evaluation数据集测试不同权重。

------

# Q9：如果RAG效果不好，你如何排查？（重点）

## 答案

我会按照链路排查：

### 1. 数据问题

检查：

- PDF解析是否正确
- 是否存在乱码
- 是否有重复header

↓

### 2. Chunk问题

检查：

- chunk size
- overlap
- chunk语义完整性

↓

### 3. Retrieval问题

检查：

- recall
- similarity score
- top-k结果

↓

### 4. Prompt问题

检查：

- context是否正确
- 是否限制模型

↓

### 5. Generation问题

检查：

- hallucination
- model能力

------

# Q10：为什么自己实现，而不是直接用LangChain/LlamaIndex？

## 答案

> 主要目的是理解RAG核心链路和工程设计。
>
> 框架可以提高开发效率，但是如果不了解Retriever、Embedding、Vector Store、Fusion这些核心组件，很难进行性能优化和问题定位。
>
> 实际生产中会根据项目需求选择框架。

------

# Q11：HybridRetriever为什么不直接写Vector和BM25逻辑？

## 答案

> 因为HybridRetriever只负责流程编排。
>
> Vector检索和BM25检索属于不同策略，应该独立。
>
> 这样未来增加新的Retriever，例如：
>
> - Elasticsearch
> - Dense Retriever
> - Multi-query Retriever
>
> 不需要修改HybridRetriever。

------

# Q12：你的项目目前还有哪些不足？

推荐回答：

> 当前版本主要完成Retrieval阶段，后续还可以继续优化：
>
> 第一，增加Evaluation模块，通过Recall@K、MRR等指标评估检索效果。
>
> 第二，引入Reranker，例如Cross Encoder，提高top-k结果质量。
>
> 第三，引入真实向量数据库，例如Milvus或ElasticSearch，提高生产能力。
>
> 第四，增加服务化部署。

------

# Q13：为什么没有做Reranker？

## 答案

> 当前版本重点是完成完整Hybrid Retrieval链路。
>
> Reranker属于Retrieval之后的精排阶段。
>
> 在企业RAG中通常流程是：
>
> Recall → Fusion → Rerank → Generation
>
> 下一阶段会加入。

------

# Q14：你的项目最大的技术难点是什么？

推荐回答：

> 最大的问题不是代码实现，而是检索效果优化。
>
> 最开始Vector Search效果不好，通过debug发现问题来自PDF解析后的文本污染，而不是embedding模型。
>
> 通过检查chunk内容、embedding结果和retrieval score定位问题，然后优化数据清洗流程。
>
> 这个过程让我理解RAG效果优化需要从数据、检索、生成整个链路分析。

------

# 架构设计专项问题（建议保留）

## Q15：为什么Pipeline不直接调用VectorStore？

答案：

> Pipeline负责业务流程编排，不应该知道底层检索细节。
>
> 通过Retriever抽象，可以替换不同检索策略。

------

## Q16：为什么Fusion是独立Strategy？

答案：

> 因为排序策略可能变化。
>
> 当前使用Weighted Fusion，未来可以替换：
>
> - RRF
> - Learning-to-Rank
> - reranking
>
> 不影响Retriever。

------

# 最后：你的项目面试定位

目前这个项目可以支撑：

✅ 初级大模型应用开发工程师
✅ RAG Engineer Junior
✅ AI Application Engineer

但是进入下一阶段前，我建议不要继续深挖这个RAG。

你的收益曲线：

```
RAG基础工程           80%
Hybrid Search        90%
继续优化RAG细节       下降
```

下一阶段更应该补：

1. **RAG Evaluation**
2. **Reranker**
3. **Agent / Tool Calling**
4. **LangGraph**
5. **部署服务化**

这才覆盖“大模型应用开发工程师”的完整能力栈。

下一步建议：

直接进入： **RAG v3.0：Evaluation + Reranker（企业RAG优化阶段）**

这个和面试价值最高。