import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP / FAISS 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import ollama
import pickle

print("===== FAISS_Persistence_RAG_LLM實驗 =====\n")

# 查詢問題（Query）
query = "我想看峽谷風景，推薦去哪？"

# 查詢向量轉換（query embedding）
q_emb = np.array([
    ollama.embeddings(model = "nomic-embed-text", prompt = query)["embedding"]
]).astype("float32")

# 讀取 FAISS index（load vector index）
loaded_index = faiss.read_index("raw.index")

# 讀取原始 documents（load documents）
with open("docs.pkl", "rb") as f:
    loaded_docs = pickle.load(f)

# Similarity Search（相似度搜尋）
D, I = loaded_index.search(q_emb, k = 1)

# 根據搜尋結果找到對應的原文（retrieve context）
context = loaded_docs[I[0][0]]

# 顯示找到的文字（display retrieved context）
print(f"找到的文字：{context}")

# LLM Generation（RAG 回答）
response = ollama.generate(
    model = "llama3.2:latest",
    prompt = f"請根據內容回答：{context}\n問題：{query}"
)

# 顯示最匹配的原文和 AI 的回答（display context and response）
print("-" * 50)
print(f"最匹配的原文：{context}")
print("-" * 50)
print(f"AI 的回答：{response['response']}")