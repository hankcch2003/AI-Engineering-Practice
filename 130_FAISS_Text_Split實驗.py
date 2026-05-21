import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("===== FAISS_Text_Split實驗 =====\n")

# Embedding model initialization（Embedding 模型初始化）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Create raw documents（建立原始文件資料）
raw_docs = [
    Document(page_content = "台灣是一個美麗的島嶼，位於亞洲東部。這裡有豐富的高山地形，最高峰是玉山。"),
    Document(page_content = "台北是台灣的首都，具有多種好吃的食物，例如牛肉麵和滷肉飯，是美食家的天堂。"),
    Document(page_content = "台南是台灣的古都，具有傳統小吃的食物，如碗粿和牛肉湯，展現了濃厚的歷史文化。"),
    Document(page_content = "宜花東地區位於台灣東部，具有豐富的風景，包括太魯閣峽谷與太平洋的海岸線。")
]

# Text splitter initialization（文本切分器）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 20,
    chunk_overlap = 5
)

# Split documents（文件切分）
docs = text_splitter.split_documents(raw_docs)

# Build FAISS vector database（建立向量資料庫）
vectorstore = FAISS.from_documents(docs, embeddings)

# Save vector database locally（儲存向量資料庫）
vectorstore.save_local("./faiss_index")

# Load vector database（載入向量資料庫）
vectorstore = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization = True
)

# Similarity search（相似度搜尋）
results = vectorstore.similarity_search("台灣有哪些山？", k = 2)

# Output result（輸出結果）
print(f"切分後的文件數量：{len(docs)}")
print("-" * 50)

for doc in results:
    print(f"搜尋結果內容：{doc.page_content}")