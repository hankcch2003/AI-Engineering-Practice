from langchain_core.runnables import RunnableLambda, RunnableSequence

print("===== 循序執行 =====")

# 轉大寫
def uppercase_func(text):
    print("1-轉換為大寫")
    return text.upper()

# 加符號
def add(text):
    print("於 2 之前加上符號")
    return text + "!!"

# 計算字數
def length_func(text):
    print("2-計算字數")
    return len(text)

# 格式化輸出
def format_result(length):
    print("3-格式化輸出")
    print()
    return f"最後結果：文字共有 {length} 個字元"

print("===== 函數轉換為元件 =====")

step1 = RunnableLambda(uppercase_func)
step1_5 = RunnableLambda(add)
step2 = RunnableLambda(length_func)
step3 = RunnableLambda(format_result)

print("===== 一個接一個方式執行（Pipeline）=====")

chain1 = step1 | step1_5 | step2 | step3
result1 = chain1.invoke("hello world")
print(result1)

print("===== 使用 RunnableSequence（標準序列）=====")

sequence1 = RunnableSequence(first = step1, middle = [step2], last = step3)
result2 = sequence1.invoke("hello world")
print(result2)

print("===== 使用 RunnableSequence（多步驟中段處理）=====")

sequence2 = RunnableSequence(first = step1, middle = [step1_5, step2], last = step3)
result3 = sequence2.invoke("hello world")
print(result3)