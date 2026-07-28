你的版本规划**方向是合理的**，但是我建议稍微调整一下版本语义，否则后面会遇到一个问题：**v1.0 通常代表“稳定可用版本”，而不是“第一个能跑的版本”**。

如果这是一个用于 GitHub / 简历展示的工程项目，我建议这样规划：

---

## 版本规划建议

### 当前版本：`v1.0.0` ✅（可以保留）

不要把它理解为“成熟生产版”，而定义为：

> First complete RAG system release

也就是：

```
v1.0.0
Basic RAG System
```

包含：

✅ PDF ingestion
✅ Text cleaning
✅ Chunking
✅ Embedding
✅ Vector DB
✅ Retrieval
✅ LLM Generation
✅ Debug Case

这个版本证明：

> 我能完整搭建一个 RAG Pipeline，并且能定位真实问题。

这是一个很好的 baseline。

---

# 下一阶段：不要直接跳 v2.0

你的：

> 开始 hybrid search 和 reranker

这个方向正确。

但是我建议拆成：

```
v1.1.0
Retrieval Optimization

v1.2.0
RAG Evaluation

v1.5.0
Advanced RAG

v2.0.0
Production RAG System
```

---

# v1.1.0 — Hybrid Search

目标：

解决 Vector Search 的天然缺陷。

现在：

```
Query
 |
Embedding
 |
Vector Similarity
 |
Top K
```

问题：

例如：

用户问：

```
退款 SLA 是多久？
```

但是文档里面：

```
Return processing period: 30 calendar days
```

语义接近，但是关键词：

```
SLA
```

不存在。

Vector Search 可能丢失。

---

加入：

```
              Query

                |
       -------------------
       |                 |
   BM25 Search     Vector Search
       |                 |
       -------------------
                |
             Fusion
                |
              Top K
```

技术：

* BM25
* Chroma similarity
* Reciprocal Rank Fusion (RRF)

版本：

```
v1.1.0

Feature:
+ Hybrid Retrieval
+ RRF ranking
+ Retrieval benchmark
```

---

# v1.2.0 — Reranker

这是非常重要的一步。

很多 RAG：

```
Retriever
   |
Top 10 chunks
   |
LLM
```

实际上：

Retriever 负责：

> 找候选

不是：

> 排最终答案

所以加入：

```
Retriever

Top 20

 |

Reranker

 |

Top 5

 |

LLM
```

例如：

模型：

* bge-reranker
* Cohere Rerank
* cross encoder

效果通常明显提升。

---

版本：

```
v1.2.0

+ Cross Encoder Reranker
+ Retrieval scoring
+ Before/After comparison
```

---

# v1.3.0 — Evaluation System

这个我建议一定加入。

因为没有 Evaluation：

你不知道：

Hybrid Search 到底有没有提升。

建立：

```
Question Dataset

        |

    RAG System

        |

 Metrics
```

指标：

## Retrieval

* Recall@K
* Precision@K
* MRR
* Hit Rate

## Generation

* Faithfulness
* Answer relevancy
* Context precision

工具：

* RAGAS
* DeepEval

---

# v1.5.0 — Advanced RAG

可以加入：

## Query Rewrite

例如：

用户：

```
退款怎么办
```

Rewrite：

```
What is the company's refund policy and process?
```

---

## Multi Query Retrieval

一个问题生成多个 query：

```
Q

 |
 +-- query1
 |
 +-- query2
 |
 +-- query3
```

---

## Metadata Filtering

例如：

```
department=finance
year=2025
```

---

# v2.0.0 — Production RAG

这个作为最终目标比较合理：

包含：

```
Production RAG System
```

能力：

```
                User

                 |

            Query Router

                 |

       ---------------------

       |                   |

   Retrieval          Tool Calling


       |

    Reranker


       |

    Context Manager


       |

       LLM


       |

 Monitoring
```

增加：

* Evaluation pipeline
* Logging
* Feedback loop
* Cost tracking
* Latency monitoring
* Deployment

---

# 所以我建议你的路线：

```
v1.0.0
✅ Basic RAG + Debug Case


v1.1.0
Hybrid Search


v1.2.0
Reranker


v1.3.0
Evaluation Framework


v1.5.0
Advanced RAG Techniques


v2.0.0
Production-grade RAG System
```

---

另外一个建议：

**不要只提交代码升级，要每个版本留下实验记录。**

比如：

```
experiments/

├── v1.0_baseline.md

├── v1.1_hybrid_search.md

├── v1.2_reranker.md

└── v1.3_evaluation.md
```

里面记录：

| Version | Method        | Recall@5 | Answer Quality |
| ------- | ------------- | -------- | -------------- |
| v1.0    | Vector Search | 0.62     | 0.70           |
| v1.1    | Hybrid        | 0.78     | 0.80           |
| v1.2    | Reranker      | 0.86     | 0.89           |

这样你的项目从：

> “我会用 LangChain 做 RAG”

升级成：

> “我设计并优化过一个 RAG 系统，并通过实验验证每次优化收益。”

这个路线非常适合作为一个长期演进项目。你现在进入的阶段，正是 RAG 工程里最有价值的部分。
