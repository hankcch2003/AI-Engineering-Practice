import sqlite3

print('===== SQLite_Select實驗 =====\n')

# Connect Database（連線資料庫）
conn = sqlite3.connect('sql1.db')

print('連線資料庫成功！')
print("-" * 50)

# Create Cursor（建立指標）
cursor = conn.cursor()

print('Cursor 建立完成！')
print("-" * 50)

# Select Data（查詢資料）
cursor.execute('SELECT * FROM students')

# Fetch All（取得所有查詢結果）
rows = cursor.fetchall()

print('查詢結果 1：')

# 顯示方式 1：完整 tuple
for row in rows:
    print(row)

print("-" * 50)
print('查詢結果 2：')

# 顯示方式 2：拆解欄位
for id, name, age, course in rows:
    print('學生：', name, '年紀:', age, '研習', course, '編號:', id)

# Select Where（條件查詢）
name1 = 'hello2'

cursor.execute(
    'SELECT * FROM students WHERE yourname = ?',
    (name1,)
)

# Fetch All（取得條件查詢結果）
rows = cursor.fetchall()

print("-" * 50)
print('條件查詢結果：')

for row in rows:
    print(row)

print("-" * 50)

# Close Database（關閉資料庫）
conn.close()

print('資料庫關閉完成！')
