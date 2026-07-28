import jieba

def chinese_tokenizer(text: str) -> list[str]:
    return list(jieba.cut(text))
