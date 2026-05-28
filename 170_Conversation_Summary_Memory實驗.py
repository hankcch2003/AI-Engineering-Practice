import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 170_Conversation_Summary_Memory實驗 =====\n")

from langchain_ollama import OllamaLLM
from langchain.memory import ConversationSummaryMemory

# LLM Model（語言模型）
llm = OllamaLLM(
    model = "llama3.2:latest",
    temperature = 0
)

# Memory（摘要記憶）
memory = ConversationSummaryMemory(llm = llm)

memory.save_context(
    {"input": "你好，我是 Python 講師"},
    {"output": "你好！很高興認識你，老師。"}
)

memory.save_context(
    {"input": "今天我們要學 RAG"},
    {"output": "好的，我準備好學習檢索增強生成了。"}
)

print("Summary Memory 內容：")
print(memory.load_memory_variables({}))