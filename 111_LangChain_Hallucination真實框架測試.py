from langchain_ollama import OllamaLLM

print("===== LLM Hallucination 測試（真實框架 + 虛構人物） =====")

# LLM model（模型初始化）
llm = OllamaLLM(model = "llama3.2:latest")

# Query（真實框架 + 假人物）
query = """
孟寶 Mengbert 是 NBA 崴孟球隊的一位籃球明星，他今天表現超好，
請問他在 NBA 這個賽季的場均得分是多少？
"""

# LLM invoke（模型呼叫）
response = llm.invoke(query)

# Output result（輸出結果）
print("模型回答：", response)