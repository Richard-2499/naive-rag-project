from llama_index.core.schema import TextNode
from openai import OpenAI

from src.schema.retrieval_result import RetrievalResult


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-v4-flash"):
        self._base_url = base_url
        self._api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
    def generate_reply(self, prompt: str) -> str:
        """
        生成答案
        Args:
            query: 用户提问
        Returns:
            str: 生成的答案
        """
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": "你是一个专业的知识库助手."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2
        )
        return response.choices[0].message.content

class Generation:
    """
    负责构造生成答案的流程
    """
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def _build_context(self, nodes: list[RetrievalResult]) -> str:
        # 把多个 textnode 对应的text 进行编号，然后再做拼接
        # contexts = "\n".join([f"【文档{i}】{text.text}" for i, text in enumerate(nodes)])

        contexts = []
        for i, node in enumerate(nodes, start=1):
            contexts.append(f"【文档{i}】\n{node.text}")

        context = "\n\n".join(contexts)
        return context

    def _build_prompt(self, query: str, context: str) -> str:
        """
        构建提示词
        """
        prompt = f"""
        【角色】 
        你是一名专业的知识库助手。
        
        【任务】
        请根据提供的相关文档，回答用户问题。如果文档中没有相关的内容，请直接返回 "没有找到相关的内容"，不要编造。
        
        【规则】
        1. 只能根据提供的参考资料回答。
        2. 如果资料中没有答案，请明确说明没有找到相关内容。
        3. 不要编造信息。
        
        【用户问题】
        {query}
        
        【相关文档】
        {context}
        
        【回答】
        
        """
        return prompt

    def generate(self, query: str, nodes: list[RetrievalResult]) -> str:
        """
        生成答案
        Args:
            query: 用户提问
        Returns:
            str: 生成的答案
        """
        context = self._build_context(nodes)
        prompt = self._build_prompt(query, context)
        return self._llm.generate_reply(prompt)



