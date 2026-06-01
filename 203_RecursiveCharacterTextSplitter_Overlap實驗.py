import warnings
import logging

# 忽略警告訊息（避免輸出過多 LangChain / Deprecation Warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

from langchain_text_splitters import RecursiveCharacterTextSplitter

print("===== 203_RecursiveCharacterTextSplitter_Overlap實驗 =====\n")

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

# Text Splitter（遞迴式切分器）
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 40,
    chunk_overlap = 10
)

# Split Text（切分文本）
chunks = splitter.split_text(text)

# Print Chunks（輸出切分結果）
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}：\n{chunk}\n{'=' * 50}")