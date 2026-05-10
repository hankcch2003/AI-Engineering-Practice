from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

print("===== 模型路由處理 =====")

# 根據文字長度決定處理方式
def route_by_length(text):
    if len(text) < 20:
        return llm.invoke(text)
    else:
        return llm.invoke(text)

# Runnable Lambda：路由處理
chain = RunnableLambda(route_by_length)

# 短文本
print("輸入：你好")
print("輸出：", chain.invoke("你好"))
print()

# 長文本
print("輸入：請詳細分析台灣AI產業未來五年發展")
print("輸出：", chain.invoke("請詳細分析台灣AI產業未來五年發展"))