from langchain_core.runnables import RunnableLambda
import time

# 定義函式：回傳問候語
def greet(name):
    return f"你好，{name} 先生小姐"

# 建立 Runnable 物件
runnable = RunnableLambda(greet)

# 設定輸入資料
names = ["Alice", "Bob", "John"]

print("===== 批量處理（batch）=====")

batch_results = runnable.batch(names)
print(batch_results)
print()

print("===== 一次性處理（invoke）=====")

invoke_result = runnable.invoke(names)
print(invoke_result)
print()

print("===== 串流處理（stream）=====")

for chunk in runnable.stream(names):
    time.sleep(0.5)
    print(chunk, end = " ", flush = True)