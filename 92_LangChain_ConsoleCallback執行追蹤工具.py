from langchain_ollama import OllamaLLM
from langchain_core.tracers.stdout import ConsoleCallbackHandler

print("===== Console Callback Trace =====")

# Callback handler
handler = ConsoleCallbackHandler()

# LLM 模型
llm = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [handler]
)

# 執行
result = llm.invoke("請用一句話形容蛋糕")
print("\n最終結果：", result)