from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

print("===== 動態 Pipeline 組裝 =====")

# 是否使用 prompt
use_prompt = True

# 基礎 passthrough（不改動輸入）
chain = RunnablePassthrough()

# 動態加入 prompt
if use_prompt:
    chain = chain.pipe(RunnableLambda(lambda x: f"請專業回答：{x}"))

# 接上 LLM
chain = chain.pipe(llm)

# 測試
print("輸入：什麼是向量資料庫")
print("結果：", chain.invoke("什麼是向量資料庫"))