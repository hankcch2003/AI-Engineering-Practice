import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 171_Custom_Summary_Memory實驗 =====\n")

from langchain_ollama import OllamaLLM
from langchain.memory import ConversationSummaryMemory
from langchain_core.prompts import PromptTemplate

# LLM Model（語言模型）
llm = OllamaLLM(
    model = "llama3.2:latest",
    temperature = 0
)

# Prompt（自訂摘要提示詞）
prompt1 = """
目前的摘要：
{summary}

新的對話片段：
{new_lines}

請用繁體中文輸出新的摘要
"""

summary_prompt = PromptTemplate(
    input_variables = ["summary", "new_lines"],
    template = prompt1
)

# Memory（自訂摘要記憶）
memory = ConversationSummaryMemory(
    llm = llm,
    prompt = summary_prompt
)

memory.save_context(
    {"input": "你好，我是 Python 講師"},
    {"output": "你好！很高興認識你，老師。"}
)

memory.save_context(
    {"input": "今天我們要學 RAG"},
    {"output": "好的，我準備好學習檢索增強生成了。"}
)

print("Custom Summary Memory 內容：")
print(memory.load_memory_variables({}))