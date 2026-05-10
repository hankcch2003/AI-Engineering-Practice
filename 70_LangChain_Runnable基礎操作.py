from langchain_core.runnables import RunnableLambda

# 定義函式：將輸入值乘以 2
def double_value(x):
    return x * 2

# 建立 Runnable 物件
runnable = RunnableLambda(double_value)

# 設定輸入值
input_value = 5

# 執行運算
result = runnable.invoke(input_value)

# 輸出結果
print("輸入值：", input_value)
print("運算結果：", result)