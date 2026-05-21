import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

print("===== PDF_RAG實驗 =====\n")

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Load PDF（讀取 PDF）
loader = PyPDFLoader(r"E:\Python\Python4-人工智慧整合開發實務\了解RNN.pdf")
pages = loader.load()

# Vector Database（向量資料庫）
db = Chroma.from_documents(
    pages,
    embedding = embedding,
    persist_directory = "./pdfdb"
)

# Query（查詢）
query = "請幫我整理LSTM綱要"

# Similarity Search（相似度搜尋）
docs = db.similarity_search(query, k = 3)

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in docs])

# 防呆（避免空資料）
if not context.strip():
    print("查無資料")
    exit()

# Prompt（提示詞）
prompt = f"""
你是嚴格的學習助理，只能根據提供資料回答。

資料：
{context}

問題：
{query}

請整理成條列式重點。
"""

# Response（模型回應）
response = llm.invoke(prompt)
print(response)