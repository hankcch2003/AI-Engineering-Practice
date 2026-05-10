from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("===== Temperature 對 LLM 幻覺影響測試 =====")

# LLM model（模型初始化）
llm = OllamaLLM(
    model = "llama3.2:latest",
    temperature = 20  # 控制隨機性（高創意／高失控，容易產生幻想）
)

# Prompt Template（防幻覺設計）
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一個基於數據的問答系統，請根據提供的資料回答問題。"
        "如果沒有相關數據，請回答『未找到相關資訊』。"
    ),
    ("user", "問題：{question}")
])

# Query（虛構問題）
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