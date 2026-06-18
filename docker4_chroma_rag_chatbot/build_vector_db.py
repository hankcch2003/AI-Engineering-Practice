import chromadb

# Vector DB builder (向量資料庫建置)
print("建置資料庫中...")
print("=" * 50)

# Initialize Chroma client (初始化 Chroma 客戶端)
client = chromadb.PersistentClient(path = "./chroma_db")

# Create collection (建立向量集合)
collection = client.get_or_create_collection(name = "demo_collection")

# Sample data (範例資料)
texts = [
    "鐵達尼號資料集包含乘客的年齡、票價與艙等資訊",
    "機器學習可以用來預測乘客是否生存",
    "年齡是影響生存機率的重要特徵"
]

# Insert data into vector DB (寫入向量資料庫)
for i, text in enumerate(texts):
    collection.add(
        documents = [text],
        ids = [f"doc_{i}"]
    )

print("資料數量：", collection.count())

# Query test (查詢測試)
query_result = collection.query(
    query_texts = ["生存預測"],
    n_results = 2
)

# Output result（輸出結果）
print("查詢結果：")
print()
print(query_result)
print("=" * 50)
print("資料庫已建置成功！")