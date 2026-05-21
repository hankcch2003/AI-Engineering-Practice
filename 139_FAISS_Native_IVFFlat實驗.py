import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP / FAISS 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import ollama

print("===== FAISS_Native_IVFFlat實驗 =====\n")

# 原始文件資料（raw documents）
raw_docs = [
    "台灣是一個美麗的島嶼，位於亞洲東部。這裡有豐富的高山地形，最高峰是玉山。",
    "台北是台灣的首都，具有多種好吃的食物，例如牛肉麵和滷肉飯，是美食家的天堂。",
    "台南是台灣的古都，具有傳統小吃的食物，如碗粿和牛肉湯，展現了濃厚的歷史文化。",
    "宜花東地區位於台灣東部，具有豐富的風景，包括太魯閣峽谷與太平洋的海岸線。"
]

# 向量化（embedding generation）
embeddings = [
    ollama.embeddings(model = "nomic-embed-text", prompt = doc)["embedding"]
    for doc in raw_docs
]

# 向量維度（vector dimension）
dim = len(embeddings[0])

# FAISS Index（IVF Flat）

# IVF 說明：
# - Inverted File Index（倒排索引）
# - 適合大規模資料
# - 需要 train + add

nlist = 2  # 分群數量（clusters）

# 建立 IVF Flat Index
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)

# 訓練 index（IVF 必須）
index.train(np.array(embeddings).astype("float32"))

# 加入向量
index.add(np.array(embeddings).astype("float32"))

# 查詢（query）
query = "我想看峽谷風景，推薦去哪？"

q_emb = np.array([
    ollama.embeddings(model = "nomic-embed-text", prompt = query)["embedding"]
]).astype("float32")

# 相似度搜尋（similarity search）
D, I = index.search(q_emb, k = 1)

# 最匹配的原文（most relevant context）
context = raw_docs[I[0][0]]

# LLM 回答（RAG）
response = ollama.generate(
    model = "llama3.2:latest",
    prompt = f"請根據內容回答：{context}\n問題：{query}"
)

# 顯示結果（display results）
print("-" * 50)
print(f"最匹配的原文：{context}")
print("-" * 50)
print(f"AI 的回答：{response['response']}")