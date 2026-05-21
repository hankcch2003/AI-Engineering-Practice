import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP / FAISS 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
import ollama
import pickle

print("===== FAISS_Persistence_Load_Only實驗 =====\n")

# 查詢內容（query text）
query = "我想看峽谷風景，推薦去哪？"

# 查詢向量轉換（query embedding）
q_emb = np.array([
    ollama.embeddings(model = "nomic-embed-text", prompt = query)["embedding"]
]).astype("float32")

# 讀取 FAISS index（load vector index）
loaded_index = faiss.read_index("raw.index")

# 讀取 documents（load raw documents）
with open("docs.pkl", "rb") as f:
    loaded_docs = pickle.load(f)

# 相似度搜尋（similarity search）
D, I = loaded_index.search(q_emb, k = 1)

# 取得最相關內容（retrieve context）
result = loaded_docs[I[0][0]]

# 顯示結果（display result）
print(f"找到的文字：{result}")