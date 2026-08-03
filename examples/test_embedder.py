# import os
# os.environ['HF_HUB_OFFLINE'] = '1'          # Hugging Face 离线模式
# os.environ['TRANSFORMERS_OFFLINE'] = '1'    # Transformers 离线模式
# from sentence_transformers import SentenceTransformer

from src.loader import *
from src.draft.config import *
from src.splitter import Splitter
from src.bge_embedder import BGEEmbedder
file_path = DATA_DIR / "raw"
docs = DocumentLoader(file_path).load_documents()

split_docs = Splitter(CHUNK_SIZE, CHUNK_OVERLAP).split_documents(docs)
# embed_model = SentenceTransformer('BAAI/bge-small-zh')
# print(type(split_docs)) # <class 'list'>
# print(type(split_docs[0])) #<class 'llama_index.core.schema.TextNode'>
texts = []
for node in split_docs:
    texts.append(node.text)
# embeddings = BGEEmbedder(config["embedding"]["model"], nomalize=True).embed([node.text for node in split_docs])
embeddings = BGEEmbedder(config["embedding"]["model"], nomalize=True).embed(texts)

print(len(embeddings))
print("dimension:", len(embeddings[0]))