from src.loader import *
from src.draft.config import *
from src.splitter import Splitter

file_path = DATA_DIR / "raw"
docs = DocumentLoader(file_path).load_documents()
# print(docs[0].text[:2000])

split_docs = Splitter(CHUNK_SIZE, CHUNK_OVERLAP).split_documents(docs)


Splitter(CHUNK_SIZE, CHUNK_OVERLAP).save_nodes_to_file(split_docs, RESULTS_DIR / "preview_split.txt")