import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 Intel MKL / OpenMP 重複載入錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

print("===== MultiFile_RAG實驗 =====\n")

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Text Splitter（文字切塊）
splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 300,
    chunk_overlap = 50
)

# Files（多檔案資料來源）
files = [
    r"E:\Python\Python4-人工智慧整合開發實務\請假制度.txt",
    r"E:\Python\Python4-人工智慧整合開發實務\薪資制度.txt",
    r"E:\Python\Python4-人工智慧整合開發實務\加班規則.txt"
]

# Load + Split（讀取 + 切塊）
all_docs = []

for file in files:
    with open(file, encoding = "utf-8") as f:
        txt = f.read()
        docs = splitter.create_documents([txt])
        all_docs.extend(docs)

# Vector Database（向量資料庫）
db = Chroma.from_documents(
    documents = all_docs,
    embedding = embedding,
    persist_directory = "./documentsdb"
)

# Query（查詢）
query = "年假有幾天？"

# Similarity Search（相似度搜尋）
results = db.similarity_search(query, k = 4)

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])

# 防呆（避免空資料）
if not context.strip():
    print("查無資料")
    exit()

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt（提示詞）
prompt = f"""
你是公司制度助理（嚴格模式）。

規則：
1. 只能根據「資料」回答
2. 若資料沒有明確答案 → 回答「查無資料」
3. 不可推測、不補充

資料：
{context}

問題：
{query}

答案：
"""

# Response（模型回應）
response = llm.invoke(prompt)
print(response)