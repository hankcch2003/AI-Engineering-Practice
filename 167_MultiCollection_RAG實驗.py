import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 167_MultiCollection_RAG實驗 =====\n")

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM

# Embedding Model（向量模型）
embedding = OllamaEmbeddings(model = "nomic-embed-text")

# Technology Collection（科技資料庫）
tech_db = Chroma.from_texts(
    texts = open("./multi_txt/科技.txt", encoding = "utf-8").read().split("\n"),
    embedding = embedding,
    collection_name = "technology",
    persist_directory = "./news_db"
)

# Finance Collection（財經資料庫）
finance_db = Chroma.from_texts(
    texts = open("./multi_txt/財經.txt", encoding = "utf-8").read().split("\n"),
    embedding = embedding,
    collection_name = "finance",
    persist_directory = "./news_db"
)

# Sports Collection（運動資料庫）
sports_db = Chroma.from_texts(
    texts = open("./multi_txt/運動.txt", encoding = "utf-8").read().split("\n"),
    embedding = embedding,
    collection_name = "sports",
    persist_directory = "./news_db"
)

print("Collection 建立完成")
print("=" * 50)

# Collection Read（讀取既有 Collection）
tech_db = Chroma(
    collection_name = "technology",
    embedding_function = embedding,
    persist_directory = "./news_db"
)

finance_db = Chroma(
    collection_name = "finance",
    embedding_function = embedding,
    persist_directory = "./news_db"
)

sports_db = Chroma(
    collection_name = "sports",
    embedding_function = embedding,
    persist_directory = "./news_db"
)

# LLM Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Query（查詢問題）
query = "台積電法說會公布營收"

# Router Prompt（分類判斷）
router_prompt = f"""
請判斷以下問題屬於哪個分類：

只能回答以下其中之一：
科技
財經
運動
其他

問題：
{query}
"""

# 分類結果
category = llm.invoke(router_prompt).strip()

print("=" * 50)
print("問題分類：", category)

# Collection Routing（資料庫路由）
if "科技" in category:
    results = tech_db.similarity_search(query, k = 3)

elif "財經" in category:
    results = finance_db.similarity_search(query, k = 3)

elif "運動" in category:
    results = sports_db.similarity_search(query, k = 3)

# 非 RAG 問題
else:
    print("=" * 50)
    print("非 RAG 問題，直接使用 LLM 回答")
    print("=" * 50)

    response = llm.invoke(query)
    print(response)
    exit()

# Context（上下文整理）
context = "\n".join([doc.page_content for doc in results])
print(context)
print("=" * 50)

print("相似文件內容：")

for i, doc in enumerate(results, start = 1):
    print()
    print(f"[{i}] {doc.page_content}")

print("=" * 50)

# Prompt（提示詞）
prompt = f"""
你是新聞助理，只能依據提供資料回答。

資料：
{context}

問題：
{query}
"""

# Response（模型輸出）
response = llm.invoke(prompt)
print("回答結果：")
print(response)