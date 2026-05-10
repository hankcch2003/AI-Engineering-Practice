import os

# 系統環境設定（System environment setup）
# 避免 Windows OpenMP / MKL 衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

print("===== LangChain_FAISSVectorDatabaseSaveLoad實驗 =====\n")

# Embedding model initialization（Embedding 模型初始化）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Create documents（建立文件資料）
docs = [
    Document(page_content = "台灣是一個美麗的島嶼"),
    Document(page_content = "台北具有多種好吃的食物"),
    Document(page_content = "台南具有傳統小吃的食物"),
    Document(page_content = "宜花東具有豐富的風景")
]

# Build FAISS vector database（建立向量資料庫）
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector database locally（儲存向量資料庫）
vectorstore.save_local("./faiss_index")

# Load vector database（載入向量資料庫）
vectorstore = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization = True)

# Similarity search（相似度搜尋）
results = vectorstore.similarity_search("台灣的地理位置", k = 2)

# Output result（輸出結果）
# 這裡會輸出與 "台灣的地理位置" 最相似的兩個文件內容
for doc in results:
    print(doc.page_content)