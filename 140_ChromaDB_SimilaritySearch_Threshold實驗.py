import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

print("===== ChromaDB_Similarity_Threshold實驗 =====\n")

# Similarity Threshold（相似度門檻）
SIMILARITY_THRESHOLD = 0.5

# Embedding Model（向量模型）
embedding_model = OllamaEmbeddings(model = "nomic-embed-text")

# Documents（文件資料）
docs = [
    Document(page_content = "志玲姐姐早餐店，一個幸福的早晨"),
    Document(page_content = "姐姐快炒，威猛好吃"),
    Document(page_content = "阿妹早午餐，永遠幸福"),
    Document(page_content = "妹妹的可愛服飾店")
]

# 建立 Chroma Vector Database
vector_db = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    collection_name = "my_collection",
    persist_directory = "./chroma_db1"
)

# 查詢（Query）
query = "志玲姐姐幸福早餐店"

# 執行相似度檢索，並獲取文件內容與對應分數
retrieved_docs_with_scores = vector_db.similarity_search_with_score(query, k = 4)

print("===== 顯示檢索到的文件內容與對應分數（分數越小代表越相似）===== \n")

# 根據相似度分數進行篩選，僅顯示分數小於或等於門檻值的文件內容與分數
for doc, score in retrieved_docs_with_scores:
    if score <= SIMILARITY_THRESHOLD:
        print(f"文件內容：{doc.page_content}")
        print(f"相似度分數：{score:.4f}")
        print("-" * 50)