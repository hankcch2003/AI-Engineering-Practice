import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

print("===== LangChain_FAISSVectorSimilaritySearch實驗 =====\n")

# Similarity threshold（相似度門檻）
# L2 距離越小代表越相似
SIMILARITY_THRESHOLD = 0.5

# Embedding model initialization（Embedding 模型初始化）
embeddings = OllamaEmbeddings(model = "nomic-embed-text")

# Create documents（建立文件資料）
docs = [
    Document(page_content = "志玲姐姐早餐店，一個幸福的早晨"),
    Document(page_content = "姐姐快炒，威猛好吃"),
    Document(page_content = "阿妹早午餐，永遠幸福"),
    Document(page_content = "妹妹的可愛服飾店")
]

# Build FAISS index（建立向量索引）
faiss_index = FAISS.from_documents(docs, embeddings)

# Query text（查詢內容）
query = "志玲姐姐幸福早餐店"

# Similarity search with score（相似度搜尋 + 分數）
retrieved_docs_with_scores = faiss_index.similarity_search_with_score(
    query, score_threshold = SIMILARITY_THRESHOLD)

# Output result（輸出結果）
# 分數越高代表越相似，通常分數會是負值，越接近 0 代表越相似
for doc, score in retrieved_docs_with_scores:
    print(f"- {doc.page_content} (分數：{score:.4f})")