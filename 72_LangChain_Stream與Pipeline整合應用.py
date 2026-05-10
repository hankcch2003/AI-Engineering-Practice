from langchain_core.runnables import RunnableLambda
import time

# 文字 stream（逐字輸出）
def greet2(text):
    for word in text.split():
        time.sleep(0.2)
        yield word + " "

# 數字 stream（逐步產生）
def num1(n):
    for i in range(n):
        time.sleep(0.2)
        yield i

# 後處理函式
def answer1(i):
    return f"處理結果：{i * 10}"

# 建立 Runnable
runnable1 = RunnableLambda(greet2)
runnable2 = RunnableLambda(num1) | RunnableLambda(answer1)

print("===== 文字 Stream =====")

text = "LLM 是大型語言模型 Runnable 將函數轉換為工具"

for chunk in runnable1.stream(text):
    print(chunk, end = "", flush = True)

print("\n")
print("===== 數字 Pipeline Stream =====")

for chunk in runnable2.stream(5):
    print(chunk, end = " ", flush = True)