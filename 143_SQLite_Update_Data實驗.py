import sqlite3

print("===== SQLite_Update_Data_實驗 =====\n")

# Connect Database（連線資料庫）
conn = sqlite3.connect("sql1.db")

print("連線資料庫成功！")
print("-" * 50)

# Create Cursor（建立指標）
cursor = conn.cursor()

print("Cursor 建立完成！")
print("-" * 50)

# Update Data（更新資料）
# 使用 WHERE 條件，避免全表更新

target_id = 1        # 指定要更新的資料 id
new_name = "Taiwan"  # 指定新的名字

cursor.execute(
    "UPDATE students SET yourname = ? WHERE id = ?",
    (new_name, target_id)
)

print(f"id = {target_id} 的資料已更新完成！")
print("-" * 50)

# Commit（確認寫入）
conn.commit()

print("資料已寫入資料庫！")
print("-" * 50)

# Close Database（關閉資料庫）
conn.close()

print("資料庫關閉完成！")