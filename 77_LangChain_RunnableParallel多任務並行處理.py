from langchain_core.runnables import RunnableLambda, RunnableParallel

print("===== 同時執行多個任務（Parallel）=====")

# 計算字串長度
def get_length(text):
    return len(text)

# 轉大寫
def get_upper(text):
    return text.upper()

# 建立並行 Runnable
parallel1 = RunnableParallel(
    length = RunnableLambda(get_length),
    upper = RunnableLambda(get_upper)
)

# 執行並行處理
result = parallel1.invoke("hello world")
print(result)