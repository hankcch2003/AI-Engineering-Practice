from langchain_core.runnables import RunnableLambda

print("===== 依序處理各種工作 =====")

# 定義轉換函式：加 3
def num1(n):
    return n + 3

# 定義轉換函式：乘 5
def num2(i):
    return i * 5

# pipeline 1：先加 3 再乘 5（順序 A → B）
runnable2 = RunnableLambda(num1) | RunnableLambda(num2)
result = runnable2.invoke(5)
print(result)

# pipeline 2：先乘 5 再加 3（順序 B → A）
runnable2 = RunnableLambda(num2) | RunnableLambda(num1)
result = runnable2.invoke(5)
print(result)