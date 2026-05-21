import sqlite3

print("===== SQLite_Insert_Data_實驗 =====\n")

# Connect Database（連線資料庫）
conn = sqlite3.connect("sql1.db")

print("連線資料庫成功！")
print("-" * 50)

# Create Cursor（建立指標）
cursor = conn.cursor()

print("Cursor 建立完成！")
print("-" * 50)

# Insert Single Data（單筆新增）
cursor.execute(
    "INSERT INTO students (yourname, age, course) VALUES (?, ?, ?)",
    ("hello", 18, "AI")
)

cursor.execute(
    "INSERT INTO students (yourname, age, course) VALUES (?, ?, ?)",
    ("hello2", 20, "AI")
)

cursor.execute(
    "INSERT INTO students (yourname, age, course) VALUES (?, ?, ?)",
    ("test3", 22, "ML")
)

print("單筆資料新增完成！")

# Commit（確認寫入）
conn.commit()

print("單筆資料已寫入資料庫！")
print("-" * 50)

# Multiple Insert（多筆新增）
studentlist = [
    ("A", 18, "NLP"),
    ("中文", 20, "神經網路"),
    ("中文A", 21, "AI OK")
]

cursor.executemany(
    "INSERT INTO students (yourname, age, course) VALUES (?, ?, ?)",
    studentlist
)

print("多筆資料新增完成！")

# Commit（確認寫入）
conn.commit()

print("多筆資料已寫入資料庫！")
print("-" * 50)

# Close Database（關閉資料庫）
conn.close()

print("資料庫關閉完成！")