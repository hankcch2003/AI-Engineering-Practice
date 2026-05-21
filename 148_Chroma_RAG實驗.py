import os
import chromadb

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("===== Chroma_RAG實驗 =====\n")

# Embedding Model（向量模型）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Vector Database（向量資料庫）
vectorstore = Chroma(
    collection_name = "my_docs",
    embedding_function = embeddings,
    persist_directory = "./chroma_db",
    client_settings = chromadb.Settings(anonymized_telemetry = False)
)

# Documents（文件資料）
docs = [
    Document(
        page_content = "台灣是一個美麗的島嶼",
        metadata = {"source": "wiki"}
    )
]

# Add Documents（加入資料）
vectorstore.add_documents(docs)

# Query（查詢問題）
query = "台灣的地理位置"

# Similarity Search（相似度搜尋）
results = vectorstore.similarity_search(query, k = 8, filter = {"source": "wiki"})

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt（提示詞）
prompt = f"""
資料：
{context}

問題：
{query}
"""

# Response（模型回應）
response = llm.invoke(prompt)
print(response)