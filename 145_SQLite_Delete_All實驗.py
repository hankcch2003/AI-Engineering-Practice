import sqlite3

print("===== SQLite_Delete_All實驗 =====\n")

# Connect Database（連線資料庫）
conn = sqlite3.connect("sql1.db")

print("連線資料庫成功！")
print("-" * 50)

# Create Cursor（建立指標）
cursor = conn.cursor()

print("Cursor 建立完成！")
print("-" * 50)

# Delete Data（無條件刪除，危險操作）
cursor.execute("DELETE FROM students")

print("所有資料已刪除完成！")
print("-" * 50)

# Commit（確認寫入）
conn.commit()

print("資料已刪除完成！")
print("-" * 50)

# Close Database（關閉資料庫）
conn.close()

print("資料庫關閉完成！")