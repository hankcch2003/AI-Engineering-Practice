import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP / FAISS 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import ollama

print("===== FAISS_Native vs LangChain比較實驗 =====\n")

print("===== 原生 FAISS（Low-level implementation）=====\n")

# 原始文件資料（raw documents）
raw_docs = [
    "台灣是一個美麗的島嶼，位於亞洲東部。這裡有豐富的高山地形，最高峰是玉山。",
    "台北是台灣的首都，具有多種好吃的食物，例如牛肉麵和滷肉飯，是美食家的天堂。",
    "台南是台灣的古都，具有傳統小吃的食物，如碗粿和牛肉湯，展現了濃厚的歷史文化。",
    "宜花東地區位於台灣東部，具有豐富的風景，包括太魯閣峽谷與太平洋的海岸線。"
]

# 向量化（embedding generation）
raw_embeddings = [
    ollama.embeddings(model = "nomic-embed-text", prompt = doc)["embedding"]
    for doc in raw_docs
]

# 向量維度（vector dimension）
dim = len(raw_embeddings[0])

# 建立 FAISS index（FAISS index construction）
index = faiss.IndexFlatL2(dim)
index.add(np.array(raw_embeddings).astype("float32"))

# 查詢內容（query text）
query = "我想看峽谷風景，推薦去哪？"

# 查詢向量轉換（query embedding）
q_emb = np.array([
    ollama.embeddings(model = "nomic-embed-text", prompt = query)["embedding"]
]).astype("float32")

# 相似度搜尋（similarity search）
D, I = index.search(q_emb, k = 1)

# 取得最相關內容（retrieve context）
context = raw_docs[I[0][0]]

# LLM 生成回答（LLM generation）
response = ollama.generate(
    model = "llama3.2:latest",
    prompt = f"請根據內容回答：{context}\n問題：{query}"
)

# Output result（輸出結果）
print(f"最匹配原文：{context}")
print("-" * 50)
print(f"AI 回答：{response['response']}")
print("-" * 50)

print("===== LangChain FAISS（High-level abstraction）=====\n")

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embedding model initialization（Embedding 模型初始化）
lc_embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Create documents（建立文件資料）
docs = [
    Document(page_content = "台灣是一個美麗的島嶼"),
    Document(page_content = "台北具有多種好吃的食物"),
    Document(page_content = "台南具有傳統小吃的食物"),
    Document(page_content = "宜花東具有豐富的風景")
]

# Build FAISS vector database（建立向量資料庫）
vectorstore = FAISS.from_documents(docs, lc_embeddings)

# Similarity search（相似度搜尋）
results = vectorstore.similarity_search("台灣的地理位置", k = 2)

# Output results（輸出結果）
for doc in results:
    print(f"搜尋結果內容：{doc.page_content}")