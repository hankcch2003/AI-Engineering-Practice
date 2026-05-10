from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("===== Strict Prompt Constraint 測試 =====")

# LLM initialization (LLM 初始化)
llm = OllamaLLM(
    model = "llama3.2:latest",
    temperature = 0  # 控制隨機性（降低亂生成）
)

# Prompt template (提示詞模板)
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "你是一個基於數據的問答系統，只允許回答『未找到相關資訊』。"
     "禁止推測、禁止補充任何額外資訊或解釋。"),
    ("user", "\n問題：{question}")
])

# Test question (測試問題)
question = "孟寶本季場均得分是多少？"

# Prompt formatting (組合提示詞)
formatted_prompt = prompt_template.format(question = question)

# Model invocation (呼叫模型)
response = llm.invoke(formatted_prompt)

# Output result (輸出結果)
print("模型回答：", response)