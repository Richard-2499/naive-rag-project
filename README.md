# RAG System

## 项目介绍

这是一个基于 Retrieval-Augmented Generation (RAG) 架构实现的知识库问答系统。

项目目标是构建一个完整的企业级 RAG Pipeline，包括：

- 文档解析
- 文本切分
- Embedding 向量化
- 向量数据库检索
- Context 拼接
- LLM 生成回答
- Debug 与问题定位

开发过程中的问题分析与优化，例如：

- PDF 文本污染导致召回失败
- Chunk 切分不合理导致上下文缺失
- Embedding 相似度不足
- Retrieval 参数调优

---

## 版本迭代
```
    v1.0
    基础 RAG Pipeline
        |
        |
    v1.2
    + Hybrid Search
    + Reranker
        |
        |
    v1.3
    + Evaluation Dataset
    + RAGAS Metrics
        |
        |
    v2.0
    Production RAG System
```


# Architecture

```
             Documents
                 |
                 v
        +----------------+
        | PDF Parser     |
        +----------------+
                 |
                 v
        +----------------+
        | Text Cleaning |
        +----------------+
                 |
                 v
        +----------------+
        | Chunk Splitter |
        +----------------+
                 |
                 v
        +----------------+
        | Embedding Model|
        +----------------+
                 |
                 v
        +----------------+
        | Vector DB      |
        +----------------+
                 |
              Query
                 |
                 v
        +----------------+
        | Retriever      |
        +----------------+
                 |
                 v
        +----------------+
        | LLM Generator  |
        +----------------+
                 |
                 v

             Answer
```

---

# Tech Stack

## Language

- Python 3.13

## LLM Framework

- 没有使用框架

## Embedding

- 本地 Embedding Model, BAAI/bge-small-zh

## Vector Database

- 存本地 json文件

## Document Processing

- PyMuPDF
- PDFPlumber

## Development

- pycharm 
- Git

---

# Pipeline

## 1. Document Loading

读取 PDF 知识文件。

流程：

```
Document
   |
   v
Parser
   |
   v
Raw Text
```

---

## 2. Text Cleaning

对原始文本进行清洗：

- 去除无关链接、日期、页眉页脚等文档污染问题


---

## 3. Chunk Splitting

将长文本切分为适合 Retrieval 的片段。

例如：

```
Document
    |
    +-- Chunk 1
    |
    +-- Chunk 2
    |
    +-- Chunk 3
```

需要平衡：

- Chunk Size
- Overlap
- Context 完整性

---

## 4. Embedding

将文本转换为向量：

```
Text
 |
 v

[0.12,0.43,0.87,...]
```

用于语义搜索。

---

## 5. Retrieval

用户输入 Query：

```
User Question

      |
      v

Vector Search

      |
      v

Relevant Documents
```

---

## 6. Generation

将检索结果加入 Prompt：

```
System Prompt

+
Retrieved Context

+
User Question

        |

        v

       LLM

        |

        v

      Answer
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>

cd rag-system
```

---

## Create Environment

```bash
python -m venv .venv

source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

创建：

```
.env
```

填写：

```env
DASHSCOPE_API_KEY=your_api_key
```

---

## Run

```bash
python main.py
```

---

# Example

## Question

```
JIT 的三个层次？
```

---

## Retrieval Result

```


```

---

## Generated Answer

```
回答:
根据提供的相关文档，JIT（Just-In-Time）的三个层次如下：

1.  **层次1：功能JIT（最小可行产品）**：不是一次性设计完整工具链，而是按需开发，例如在Day 1开发格式校验脚本，Day 5开发统计分析脚本，Day 10开发导出功能，无冗余功能。

2.  **层次2：质量JIT（梯度收紧）**：不是所有数据都要求95%准确率，而是分阶段提升质量，例如P0阶段达到60%准确率，P1阶段达到80%，P2阶段达到90%，P3阶段达到97%，逐轮提升。

3.  **层次3：知识JIT（按需沉淀）**：不是项目结束后写一份完整文档，而是随项目演化逐步沉淀，例如Week 1写冷启动文档，Week 2写本体设计文档，Week 4写反思方法文档，Week 8写完整META技能文档。

```

---

# Debug Case

## PDF文本污染导致召回失败

### 问题现象

用户输入：

```

```



Retrieval 返回：

```


```

---

## 问题定位过程

### Step 1: 检查 Retrieval

首先确认：

- Vector DB 是否正常
- Query Embedding 是否生成
- Top-K 是否返回结果

发现：

```


```

---

### Step 2: 查看原始 PDF 文本


---

### Step 3: 定位原因

原因：


---

## 解决方案

### 方案 1：增加文本清洗



---

### 方案 2：优化 PDF Parser


---

### 方案 3：增加 Debug Pipeline



---

# Future Improvements

[//]: # (- Hybrid Search &#40;BM25 + Vector Search&#41;)

[//]: # (- Reranker)

[//]: # (- Query Rewrite)

[//]: # (- Evaluation Dataset)

[//]: # (- RAG Benchmark)

---

# Author

RAG System Engineering Practice