import sqlite3

print("===== SQLite_Update_Where實驗 =====\n")

# Connect Database（連線資料庫）
conn = sqlite3.connect("sql1.db")

print("連線資料庫成功！")
print("-" * 50)

# Create Cursor（建立指標）
cursor = conn.cursor()

print("Cursor 建立完成！")
print("-" * 50)

# Update Data（條件更新）
newname = "Taiwan"
newcourse = "Python"

cursor.execute(
    "UPDATE students SET course = ? WHERE yourname = ?",
    (newcourse, newname)
)

print("資料更新完成！")
print("-" * 50)

# Commit（確認寫入）
conn.commit()

print("資料已寫入資料庫！")
print("-" * 50)

# Close Database（關閉資料庫）
conn.close()

print("資料庫關閉完成！")