from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("===== Context Aware Prompt 實驗 =====")

# LLM initialization (LLM 初始化)
llm = OllamaLLM(
    model = "llama3.2:latest",
    temperature = 0  # 控制隨機性（降低亂生成）
)

# Prompt template (提示詞模板)
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     "你是一個基於數據的問答系統，請根據提供的資料回答問題。"
     "如果沒有相關數據，請只回答『未找到相關資訊』，"
     "不得添加任何額外資訊或解釋，也不要推測數字。"),
    ("user", "數據來源：\n{context}\n\n問題：{question}")
])

# Context (資料來源)
context = "孟寶 Mengbert 在 NBA 這個賽季的場均得分為 30.1 分，助攻為 5.7 次。"

# Test question (測試問題)
question = '''孟寶 Mengbert 是 NBA 崴孟球隊的一位籃球明星，他今天表現超好，
請問他在 NBA 這個賽季的場均得分是多少？'''

# Prompt formatting (組合提示詞)
formatted_prompt = prompt_template.format(
    context = context,
    question = question
)

# Model invocation (呼叫模型)
response = llm.invoke(formatted_prompt)

# Output result (輸出結果)
print("模型回答：", response)