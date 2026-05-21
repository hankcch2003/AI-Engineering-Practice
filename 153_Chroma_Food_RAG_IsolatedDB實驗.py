import os
import chromadb

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("===== Chroma_Food_RAG_IsolatedDB實驗 =====\n")

# Embedding Model（向量模型）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Vector Database（向量資料庫）
vectorstore = Chroma(
    collection_name = "my_docs",
    embedding_function = embeddings,
    persist_directory = "./food_chroma_db",
    client_settings = chromadb.Settings(anonymized_telemetry = False)
)

# Documents（資料擴充）
docs = [
    Document(page_content = "台灣是一個美麗的島嶼", metadata = {"source": "wiki"}),
    Document(page_content = "台灣位於亞洲東部、太平洋西岸。", metadata = {"source": "wiki"}),
    Document(page_content = "台灣北臨東海、南臨巴士海峽。", metadata = {"source": "report"}),
    Document(
        page_content = "台灣人對於美食永遠不會統一意見",
        metadata = {
            "source": "food",
            "page": 1,
            "author": "課程",
            "date": "2026-05-14"
        }
    ),
    Document(
        page_content = "煎餃的經典口味有高麗菜豬肉、韭菜豬肉、玉米豬肉/鮮蝦",
        metadata = {"source": "food", "author": "課程"}
    ),
    Document(
        page_content = "煎餃的特色創意口味有 打拋豬肉、紅咖哩、綠咖哩、章魚燒（柴魚片+美乃滋）、炙燒起司玉米、明太子、青蔥味噌",
        metadata = {"source": "food", "author": "課程"}
    ),
    Document(
        page_content = "煎餃食用建議除了原味，搭配東泉辣椒醬、特製醬汁鹹甜微辣更夠味",
        metadata = {"source": "food", "author": "課程"}
    )
]

# Add Documents（加入資料）
vectorstore.add_documents(docs)

# Query（查詢）
question1 = "煎餃的泰式風味"

# Similarity Search（相似度搜尋）
results = vectorstore.similarity_search(question1, k = 8, filter = {"source": "food"})

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt（嚴謹模式）
prompt = f"""
你是公司制度助理，只能依據提供資料回答。
資料：
{context}

問題：
{question1}
"""

# Response（模型回應）
response = llm.invoke(prompt)
print(response)