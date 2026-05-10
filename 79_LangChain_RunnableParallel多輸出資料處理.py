from langchain_core.runnables import RunnableParallel, RunnableLambda

# 轉大寫
def uppercase_func(data):
    return data["input_text"].upper()

# 計算字數
def length_func(data):
    return len(data["input_text"])

# Runnable Lambda：大寫處理
upper_step = RunnableLambda(uppercase_func)

# Runnable Lambda：長度計算
length_step = RunnableLambda(length_func)

# RunnableParallel：同時執行多個任務
map_chain = RunnableParallel(
    original = RunnableLambda(lambda x: x["input_text"]),
    upper = upper_step,
    count = length_step
)

# 執行 parallel chain
final_data = map_chain.invoke({"input_text": "hello langChain"})

# 輸出結果
print(final_data)