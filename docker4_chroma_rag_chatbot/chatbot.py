import chromadb

# Vector DB chatbot (向量資料庫問答系統)
print("Chroma Chatbot 啟動中...")
print("=" * 50)

# Initialize Chroma client (初始化 Chroma 客戶端)
client = chromadb.PersistentClient(path = "./chroma_db")

# Get collection (取得向量集合)
collection = client.get_or_create_collection(name = "demo_collection")

# 如果 collection 是空的就補資料
if collection.count() == 0:
    texts = [
        "鐵達尼號資料集包含乘客的年齡、票價與艙等資訊",
        "機器學習可以用來預測乘客是否生存",
        "年齡是影響生存機率的重要特徵"
    ]

    for i, text in enumerate(texts):
        collection.add(
            documents = [text],
            ids = [f"doc_{i}"]
        )

# User input (使用者輸入)
query = input("請輸入問題：")
print("=" * 50)

# Query vector DB (查詢向量資料庫)
results = collection.query(
    query_texts = [query],
    n_results = 2
)

# Output results (輸出結果)
print("回答結果：")
print()

if results["documents"] and len(results["documents"][0]) > 0:
    for i, doc in enumerate(results["documents"][0]):
        print(f"{i + 1}. {doc}")
else:
    print("沒有找到相關資料")

print("=" * 50)
print("查詢完成！")