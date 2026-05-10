from langchain_core.runnables import RunnableLambda
import time

# 文字轉換 stream（中文數字 → 阿拉伯數字）
def greet2(text):
    to_num = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5"
    }

    for word in text.split():
        time.sleep(0.5)
        converted = to_num.get(word, word)
        yield converted + " "

# 建立 Runnable 物件
runnable1 = RunnableLambda(greet2)

print("===== 連續性 Stream 資料轉換處理 =====")

print("假設輸入：一二三 → 轉換為 123")

text = "LLM 是大型語言模型，一 二 三，Runnable 將函數轉換為工具"

# 逐步串流輸出處理結果
for chunk in runnable1.stream(text):
    print(chunk, end = "", flush = True)