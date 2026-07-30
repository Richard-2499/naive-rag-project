from src.loader import *
from src.splitter import Splitter
from config.config_loader import load_config
config = load_config()
file_path = config["paths"]["raw_data"]
docs = DocumentLoader(file_path).load_documents()
# print(docs[0].text[:2000])

split_docs = Splitter(config["splitter"]["chunk_size"], config["splitter"]["chunk_overlap"]).split_documents(docs)
# print("node_id: ",split_docs[1].node_id)
# print("id_ : ",split_docs[1].id_)
# print("Node ID: ", split_docs[1].dict())

Splitter(
    config["splitter"]["chunk_size"],
    config["splitter"]["chunk_overlap"]
    ).save_nodes_to_file(
        split_docs,
        config["paths"]["chunks_store"]
    )