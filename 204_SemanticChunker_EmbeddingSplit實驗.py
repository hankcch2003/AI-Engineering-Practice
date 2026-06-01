import warnings
import logging

# 忽略警告訊息（避免輸出過多 warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

print("===== 204_SemanticChunker_EmbeddingSplit實驗 =====\n")

# Text Data（測試文本）
text = """第一章：機器學習基礎。
機器學習是人工智慧的一個分支。它讓電腦能從資料中學習規律。

第二章：深度學習與影像辨識。
卷積神經網路（CNN）非常適合處理影像資料。
透過多層特徵提取，它可以自動辨識出物件。

第三章：植物照顧。
辦公室的植物栽種也需要注意。
特別是多肉植物與空氣鳳梨，

在照顧時必須嚴格控制澆水頻率與環境通風，
如果環境過於潮濕，容易導致根部腐爛。

蘭花在栽培時需要注意通風與濕度控制。
特別是蝴蝶蘭，過度澆水容易導致根部腐爛。
"""

# Embedding Model（向量模型）
print("正在初始化 Embedding 模型...")
print("=" * 50)

embeddings = OllamaEmbeddings(
    model = "nomic-embed-text",
    base_url = "http://localhost:11434"
)

# Semantic Chunker（語意切分器）
print("建立 Semantic Chunker...")
print("=" * 50)

text_splitter = SemanticChunker(embeddings,

    # 中文句子切分規則
    sentence_split_regex = r'(?<=[。？！\n])',

    # 斷點策略（標準差）
    breakpoint_threshold_type = "standard_deviation",

    # 控制切分敏感度（越小越細）
    breakpoint_threshold_amount = 1.0
)

# Split Text（語意切分）
print("開始語意切割...")
print("=" * 50)

chunks = text_splitter.split_text(text)

print(f"共切分為 {len(chunks)} 個區塊")
print("=" * 50)

# Print Chunks（輸出結果）
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}：")
    print(chunk.strip())
    print("=" * 50)