import sqlite3

# Connect Database（連線資料庫）
# 若資料庫不存在，SQLite 會自動建立資料庫檔案
conn = sqlite3.connect("sql1.db")

print("===== SQLite_Create_Table實驗 =====\n")

print("連線資料庫成功！")
print("資料庫名稱：sql1.db")
print("-" * 50)

# Create Cursor（建立指標）
# Cursor 用來執行 SQL 指令
cursor = conn.cursor()

print("建立 Cursor 成功！")
print("-" * 50)

# Create Table（建立資料表）
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    yourname TEXT NOT NULL,
    age INTEGER,
    course TEXT
)
""")

print("students 資料表建立完成！")
print("-" * 50)

# Commit（確認寫入）
conn.commit()

print("資料表建立成功！")
print("-" * 50)

# Close Database（關閉資料庫）
conn.close()

print("資料庫關閉完成！")