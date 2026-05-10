from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

print("===== 動態 Pipeline 迴圈組裝 =====")

# 初始 chain
chain = RunnablePassthrough()

# 多段處理流程
processors = [
    lambda x: x.upper(),
    lambda x: f"問題：{x}",
    lambda x: f"{x}，請回答"
]

# 逐步組裝 pipeline
for p in processors:
    chain = chain.pipe(RunnableLambda(p))

# 接 LLM
chain = chain.pipe(llm)

# 測試
print("輸入：Python")
print("結果：", chain.invoke("Python"))