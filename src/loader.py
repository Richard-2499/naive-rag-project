from llama_index.core import SimpleDirectoryReader
from pathlib import Path

from llama_index.core.schema import Document

from src.logger import get_logger
logger = get_logger(__name__)


class DocumentLoader:
    """
    负责加载文档，不负责解析、切分和向量化
    """
    SUPPORTED_EXTENSIONS = [".pdf", ".txt"]
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
    def load_documents(self) -> list[Document]:
        '''
        加载指定目录下所有文档
        Returns:
             List[Document]: LlamaIndex 的Document 对象列表
        '''
        logger.info(f"📂 从{self.data_dir} 加载文档 ...")
        documents = SimpleDirectoryReader(
            input_dir = str(self.data_dir),
            required_exts = self.SUPPORTED_EXTENSIONS,
            recursive = False,
        ).load_data()
        for doc in documents:
            cleaned_text = self._clean_text(doc.text)
            doc.set_content(cleaned_text)

        # print("doc type: ",type(documents))  #  doc type:  <class 'list'>
        # print("doc length: ", len(documents))  # doc length:  1
        # print(documents[:1])

        logger.info(f"✅ 加载完成，共 {len(documents)} 个文档")
        # logger.info(f"📄 文档列表：{documents}")
        logger.info(f"📄 总字符数：{sum(len(doc.text) for doc in documents)}")
        return documents
    @staticmethod
    def _clean_text(text: str) -> str:
        """
        清理 PDF 抽取产生的无关内容
        """
        lines = text.splitlines()
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "mp.weixin.qq.com" in line:
                continue
            if "好东西都是总结出来的" in line:
                continue
            if "上海Original 01fish" in line:
                continue
            if "2w 字" in line:
                continue
            if "NetEngine 8000 X" in line:
                continue
            if "版权所有 © 华为技术有限公司" in line:
                continue
            if "NetEngine 8000E X 系列" in line:
                continue
            if "2024-03-30" in line:
                continue
            if "配置指南 1 局域网与城域" in line:
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

