import warnings
import logging

# 忽略警告訊息（避免輸出過多 warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

import numpy as np
import ollama
from rank_bm25 import BM25Okapi
import jieba

print("===== 207_Hybrid_RAG實驗 =====\n")

# Documents（資料）
documents = [
    "【Python 安裝教學】在 Windows 環境下安裝 Python 3.11。",
    "【Ollama 執行指南】Llama 3.2 需要 4GB 記憶體，建議 8GB 以上。",
    "【機器學習入門】線性回歸與 Scikit-Learn。",
    "【Llama 3.2 介紹】支援邊緣運算與多模態輸入。"
]

# Vector Search（向量搜尋）
print("===== [Step 1] Vector Search =====\n")

doc_vectors = []

for doc in documents:
    res = ollama.embeddings(model = "nomic-embed-text", prompt = doc)
    doc_vectors.append(res["embedding"])

doc_vectors = np.array(doc_vectors)

query = "Llama3.2 規格與記憶體需求"

query_vec = np.array(
    ollama.embeddings(
        model = "nomic-embed-text",
        prompt = query
    )["embedding"]
)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

vector_scores = [
    cosine_similarity(query_vec, doc_vec)
    for doc_vec in doc_vectors
]

vector_ranking = np.argsort(vector_scores)[::-1]

print("Vector Ranking:", vector_ranking)
print("=" * 50)

# BM25 Search（關鍵字搜尋）
print("===== [Step 2] BM25 Search =====\n")

tokenized_query = list(jieba.cut(query))
tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]

bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(tokenized_query)

keyword_ranking = np.argsort(bm25_scores)[::-1]

print()
print("BM25 Ranking:", keyword_ranking)
print("=" * 50)

# RRF Fusion（分數融合）
print("===== [Step 3] RRF Fusion =====\n")

k = 60
rrf_scores = np.zeros(len(documents))

for rank, idx in enumerate(vector_ranking):
    rrf_scores[idx] += 1 / (k + rank + 1)

for rank, idx in enumerate(keyword_ranking):
    rrf_scores[idx] += 1 / (k + rank + 1)

best_idx = np.argmax(rrf_scores)
hybrid_context = documents[best_idx]

print("Best Document Index:", best_idx)
print("Context:", hybrid_context)
print("=" * 50)

# LLM Generation（生成答案）
print("===== [Step 4] LLM Answer =====\n")

prompt = f"""
你是一個精準 AI 助理。
請根據參考資料回答問題，若沒有資訊請回答不知道。

【參考資料】
{hybrid_context}

【問題】
{query}
"""

response = ollama.generate(
    model = "llama3.2:latest",
    prompt = prompt
)

print(response["response"])