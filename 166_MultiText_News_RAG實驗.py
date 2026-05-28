import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 166_MultiText_RAG實驗 =====\n")

import glob
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Text Splitter（文本切塊工具）
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

# 讀取資料（多文本資料夾）
files = glob.glob("./multi_txt/*.txt")

all_docs = []

# 讀取 + 切塊
for file in files:
    print("讀取檔案：", file)

    with open(file, encoding = "utf-8") as f:
        txt = f.read()
        docs = splitter.create_documents(
            [txt],
            metadatas = [{"source": file}]
        )
        all_docs.extend(docs)

# 統計文件數量
print("=" * 50)
print(f"總文件數量：{len(all_docs)}")

# Vector Database（向量資料庫）
db = Chroma.from_documents(
    documents = all_docs,
    embedding = embedding,
    persist_directory = "./newsdb"
)

# Query（查詢問題）
query = "籃球"

# Similarity Search（向量檢索）
results = db.similarity_search(query, k = 50)

# Context（上下文整理）
context = "\n".join([
    doc.page_content
    for doc in results
])

print("=" * 50)
print("相似文件內容：")
print()
print(context)
print("=" * 50)

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt（提示詞）
prompt = f"""
你是一個新聞資料助理，只能依據提供資料回答。

規則：
1. 只能使用提供的資料
2. 不可自行補充外部資訊
3. 若沒有答案請回答：查無資料

資料：
{context}

問題：
{query}
"""

# Response（模型輸出）
response = llm.invoke(prompt)
print("回答結果：")
print(response)