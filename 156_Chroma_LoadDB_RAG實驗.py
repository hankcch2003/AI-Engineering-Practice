import os
import chromadb

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

print("===== Chroma_LoadDB_RAG實驗 =====\n")

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Load Vector Database（載入向量資料庫）
db = Chroma(
    embedding_function = embedding,
    persist_directory = "./text_db",
    client_settings = chromadb.Settings(anonymized_telemetry = False)
)

# Query（查詢）
query = "年假有幾天？"

# Similarity Search（相似度搜尋）
results = db.similarity_search(query, k = 2)

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt（回答生成）
prompt = f"""
根據以下資料回答：
{context}

問題：
{query}
"""

# Response（模型回應）
response = llm.invoke(prompt)
print(response)