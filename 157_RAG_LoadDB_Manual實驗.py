import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma

print("===== RAG_LoadDB_Manual實驗 =====\n")

# Embedding Model（向量模型）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Vector Database（向量資料庫）
vectorstore = Chroma(
    embedding_function = embeddings,
    persist_directory = "./text_db"
)

# Retriever（檢索器）
retriever = vectorstore.as_retriever(search_kwargs = {"k": 3})

# Query（查詢）
query = "年假有幾天？"

# Retrieve Documents（取得文件）
docs = retriever.invoke(query)

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in docs])

# 如果沒有資料，則輸出「查無資料」並結束程式
if not context.strip():
    print("查無資料")
    exit()

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt（提示詞）
prompt = f"""
你是嚴格的制度助理。

規則：
1. 只能使用提供的資料回答
2. 如果資料沒有明確答案，請只輸出：查無資料
3. 不可以猜測、不可以補充

資料：
{context}

問題：
{query}

答案：
"""

# Response（模型回應）
response = llm.invoke(prompt)
print(response)