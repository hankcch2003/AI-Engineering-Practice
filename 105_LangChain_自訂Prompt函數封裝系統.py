from langchain_ollama import OllamaLLM

print("===== 自訂 Prompt 函數封裝系統 =====")

# initialize LLM（模型初始化）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt Wrapper Function（提示詞封裝函數）
# 用於統一 prompt 風格並重複使用
def trump_invoke(text):

    # prompt engineering（提示詞設計）
    prompt = (
        "請以川普語氣，自信、誇張、口語化回答："
        "請用一句話形容 " + text
    )

    # LLM Invoke（模型呼叫）
    return llm.invoke(prompt)

# test cases（測試資料）
result = trump_invoke("蛋糕")
print("模型回答1：", result)

result = trump_invoke("英國茶")
print()
print("模型回答2：", result)

result = trump_invoke("巧克力")
print()
print("模型回答3：", result)

result = trump_invoke("白宮")
print()
print("模型回答4：", result)

result = trump_invoke("溫莎古堡")
print()
print("模型回答5：", result)