from langchain_core.runnables import RunnableLambda
import time

# 文字 stream（逐字輸出）
def greet2(text):
    for word in text.split():
        time.sleep(0.5)
        yield word + " "

# 建立 Runnable 物件
runnable1 = RunnableLambda(greet2)

print("===== 連續性 Stream 處理 =====")

text = "LLM 是大型語言模型，Runnable 將函數轉換為工具一步二步三步向前走"

# 逐步串流輸出處理結果
for chunk in runnable1.stream(text):
    print(chunk, end = "", flush = True)