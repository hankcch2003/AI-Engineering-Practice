import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

print("===== FAISS_RAG實驗 =====\n")

# Embedding model initialization（Embedding 模型初始化）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Load FAISS vector database（載入向量資料庫）
vectorstore = FAISS.load_local(
    "./faiss_index2",
    embeddings,
    allow_dangerous_deserialization = True
)

# Query text（查詢內容）
query = "我想看峽谷風景，推薦去哪？"

# Similarity search（相似度搜尋）
results = vectorstore.similarity_search(query, k = 2)

# Collect context（整理上下文）
context_list = []

for doc in results:
    print(f"搜尋結果內容：{doc.page_content}")
    context_list.append(doc.page_content)

# LLM generation（交給模型生成答案）
response = ollama.generate(
    model = "llama3.2:latest",
    prompt = f"請根據以下內容回答：{context_list}\n問題：{query}"
)

# Output result（輸出結果）
print("-" * 50)
print(f"最匹配原文：{context_list}")
print("-" * 50)
print(f"AI 回答：{response['response']}")