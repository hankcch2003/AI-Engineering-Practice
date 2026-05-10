from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# LLM 模型
small_llm = OllamaLLM(model = "phi3:mini")
big_llm = OllamaLLM(model = "llama3.2:latest")

print("===== 雙模型條件式 Pipeline 控制 =====")

# 是否使用 prompt
use_prompt = False

# 基礎 passthrough
chain = RunnablePassthrough()

# 動態加入 prompt
if use_prompt:
    chain = chain.pipe(RunnableLambda(lambda x: f"請專業回答：{x}"))

# 模型選擇邏輯（雙模型切換）
def model_router(x):
    if len(x) < 20:
        print("[系統訊息]（使用小模型）")
        return small_llm.invoke(x)
    else:
        print("[系統訊息]（使用大模型）")
        return big_llm.invoke(x)

# 接模型 router
chain = chain.pipe(model_router)

# 測試
print("輸入：什麼是向量資料庫")
print("結果：", chain.invoke("什麼是向量資料庫"))