from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

print("===== PromptTemplate + Chain Pipeline 系統 =====")

# initialize LLM（模型初始化）
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt Template（提示詞模板）
# 用於結構化 prompt，提升可重用性
prompt = PromptTemplate.from_template(
"""
請以川普語氣回答：
特徵：
- 自信
- 誇張
- 煽動性
- 口語化
- 使用最高級形容詞

請用一句話形容：
問題：{question}
"""
)

# Chain Pipeline（鏈式結構）
chain = prompt | llm

# Test cases（測試資料）
result = chain.invoke({"question": "蛋糕"})
print("模型回答1：", result)

result = chain.invoke({"question": "英國茶"})
print()
print("模型回答2：", result)

result = chain.invoke({"question": "巧克力"})
print()
print("模型回答3：", result)

result = chain.invoke({"question": "白宮"})
print()
print("模型回答4：", result)

result = chain.invoke({"question": "溫莎古堡"})
print()
print("模型回答5：", result)