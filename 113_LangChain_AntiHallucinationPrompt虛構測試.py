from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("===== Anti-Hallucination Prompt（無真實背景測試） =====")

# LLM model（模型初始化）
llm = OllamaLLM(
    model = "llama3.2:latest",
    temperature = 10  # 控制隨機性（中等創意與穩定性平衡）
)

# Prompt Template（防幻覺設計 / guardrail）
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一個基於數據的問答系統，請根據提供的資料回答問題。"
        "如果沒有相關數據，請回答『未找到相關資訊』，不要推測數字。"
    ),
    ("user", "問題：{question}")
])

# Query（完全虛構問題）
question = """
孟寶 Mengbert 是崴孟球隊的一位籃球明星，他今天表現超好，
請問他在這個賽季的場均得分是多少？
"""

# Prompt formatting（提示詞組裝）
formatted_prompt = prompt_template.format(question = question)

# LLM invoke（模型呼叫）
response = llm.invoke(formatted_prompt)

# Output result（輸出結果）
print("模型回答：", response)