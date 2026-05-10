from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt 模板
prompt = PromptTemplate.from_template("請用20字以內描述：{text}")

# 產生 Prompt
prompt_value = prompt.invoke({"text": "貓咪"})
print("Prompt 輸出：", prompt_value)

# LLM 回應
response = llm.invoke(prompt_value)
print("LLM 回應：", response)