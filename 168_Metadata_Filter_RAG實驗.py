import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 168_Metadata_Filter_RAG實驗 =====\n")

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Documents（建立文件資料）
docs = []

# 讀取科技資料
with open("./multi_txt/科技.txt", encoding = "utf-8") as f:
    lines = f.read().split("\n")

    for line in lines:
        if line.strip():
            docs.append(
                Document(
                    page_content = line,
                    metadata = {
                        "category": "technology",
                        "source": "科技.txt"
                    }
                )
            )

# 讀取財經資料
with open("./multi_txt/財經.txt", encoding = "utf-8") as f:
    lines = f.read().split("\n")

    for line in lines:
        if line.strip():
            docs.append(
                Document(
                    page_content = line,
                    metadata = {
                        "category": "finance",
                        "source": "財經.txt"
                    }
                )
            )

# 讀取運動資料
with open("./multi_txt/運動.txt", encoding = "utf-8") as f:
    lines = f.read().split("\n")

    for line in lines:
        if line.strip():
            docs.append(
                Document(
                    page_content = line,
                    metadata = {
                        "category": "sports",
                        "source": "運動.txt"
                    }
                )
            )

print("文件總數量：", len(docs))

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Vector Database（向量資料庫）
db = Chroma.from_documents(
    documents = docs,
    embedding = embedding,
    collection_name = "news",
    persist_directory = "./news2_db"
)

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Query（查詢問題）
query = "黃金價格"

# Router Prompt（分類路由）
router_prompt = f"""
請判斷問題屬於哪個分類：

只能回答：
technology
finance
sports
other

問題：
{query}
"""

# 分類結果
category = llm.invoke(router_prompt).strip()

print("=" * 50)
print("分類結果：", category)

# Metadata Filter（Metadata 過濾）
if category != "other":
    results = db.similarity_search(query, k = 3, filter = {"category": category})

    print("=" * 50)
    print("相似文件內容：")

    for i, doc in enumerate(results, start = 1):
        print("=" * 50)
        print(f"[{i}] {doc.page_content}")

else:
    print("=" * 50)
    print("非 RAG 問題，直接使用 LLM 回答")

    response = llm.invoke(query)
    print(response)
    exit()

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])
print(context)
print("=" * 50)

# Prompt（提示詞）
prompt = f"""
你是新聞助理。

請依據資料回答：

{context}

問題：
{query}
"""

# Response（模型輸出）
response = llm.invoke(prompt)
print("回答結果：")
print(response)