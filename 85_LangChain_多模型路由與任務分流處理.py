from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda

# LLM 模型
small_llm = OllamaLLM(model = "phi3:mini")
big_llm = OllamaLLM(model = "llama3.2:latest")

print("===== 多模型路由處理 =====")

# 路由規則
def smart_router(text):
    if "翻譯" in text:
        print("[系統訊息] 任務：翻譯（使用小模型）")
        return small_llm.invoke(f"請將以下內容翻譯成中文：{text}")
    else:
        print("[系統訊息] 任務：問答（使用大模型）")
        return big_llm.invoke(f"請以 AI 領域說明：{text}")

# Runnable Lambda：路由處理
chain = RunnableLambda(smart_router)

# 測試問題
questions = [
    "什麼是 AI？",
    "請翻譯：LangChain makes it easy to build LLM applications.",
    "什麼是 RAG？"
]

# 輸出結果
for q in questions:
    print("問題：", q)
    result = chain.invoke(q)
    print("結果：", result)
    print("-" * 30)