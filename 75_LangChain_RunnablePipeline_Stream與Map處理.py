from langchain_core.runnables import RunnableLambda
import time

# 數字 stream（逐步產生）
def num1(n):
    for i in range(n):
        time.sleep(0.5)
        yield i

# 後處理函式（map）
def answer1(i):
    print("i =", i)
    return f"處理結果：{i * 10}"

# 建立 Pipeline
runnable2 = RunnableLambda(num1) | RunnableLambda(answer1)

print("===== 連續性 Pipeline Stream 處理 =====")

# 逐步串流輸出處理結果
for chunk in runnable2.stream(5):
    print(chunk, end = " ", flush = True)