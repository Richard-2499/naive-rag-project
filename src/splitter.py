from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from pathlib import Path

from llama_index.core.schema import TextNode
from src.logger import get_logger
logger = get_logger(__name__)

class Splitter:
    """
    负责把文档切成更小的块
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: list[Document]) -> list[TextNode]:
        """
        切分分档
        Args:
            documents: 要切分的文档完整路径
        Returns:
            list[Document]
        """
        # logger.info(f"Document 数量: {len(documents)}")  # Document 数量: 55
        # logger.info(f"Document 长度: {len(documents[1].text)}")  # Document 长度: 663
        # logger.info(documents[0].text[:500])
        # logger.info(documents[0].metadata)

        nodes = self._parser.get_nodes_from_documents(documents) # list[TextNode]
        # logger.info(type(nodes[0]))
        # logger.info(vars(nodes[0]).keys())
        '''
        dict_keys(['id_', 'embedding', 'metadata', 'excluded_embed_metadata_keys', 'excluded_llm_metadata_keys', 'relationships', 
        'metadata_template', 'metadata_separator', 'text', 'mimetype', 'start_char_idx', 'end_char_idx', 'text_template'])
        '''
        # logger.info(list(vars(nodes[0]).keys()))
        # exit(0)
        total_chars = sum(len(node.text) for node in nodes)
        avg_chars = total_chars / len(nodes) if nodes else 0
        logger.info(f"✅ 文档切分完成，共 {len(nodes)} 个节点，平均长度 {avg_chars:.2f} 个字符")
        return nodes

    def save_nodes_to_file(self, nodes: list[TextNode], output_path: Path):
        """
        查看文档切割是否与预期相符
        Args:
            nodes: 切分好的文本块列表
        Returns:
            None
        """
        with output_path.open("w", encoding="utf-8") as f:
            f.write(f"切分预览 - chunk_size = {self._chunk_size}, overlap = {self._chunk_overlap}")
            f.write("\n\n"+"=" * 80 + "\n\n")

            for i, node in enumerate(nodes, 1):
                f.write(f"节点 {i} 长度 {len(node.text)} 个字符:\n")
                f.write(node.text[:1000])
                if len(node.text)>1000:
                    f.write("...")
                f.write("\n\n" + "-" * 80 + "\n\n")
        logger.info(f"✅ 节点保存完成，保存路径：{output_path}")


