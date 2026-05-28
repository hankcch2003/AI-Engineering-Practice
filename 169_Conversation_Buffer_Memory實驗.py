import os

# 系統環境設定（System environment setup）
# 避免 Windows 上 MKL / OpenMP 多執行緒衝突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("===== 169_ConversationBufferMemory實驗 =====\n")

from langchain.memory import ConversationBufferMemory

# Memory（對話記憶）
memory = ConversationBufferMemory()

memory.save_context(
    {"input": "你好，我是 Python 講師"},
    {"output": "你好！很高興認識你，老師。"}
)

memory.save_context(
    {"input": "今天我們要學 RAG"},
    {"output": "好的，我準備好學習檢索增強生成了。"}
)

print("Memory內容：")
print(memory.load_memory_variables({}))