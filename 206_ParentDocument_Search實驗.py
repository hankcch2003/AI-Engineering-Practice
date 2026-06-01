import warnings
import logging
import ollama

# 忽略警告訊息（避免輸出過多 warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

print("===== 206_ParentDocument_Search實驗 =====\n")

# Parent Document（父文件）
parent_document = """
【Ollama 執行指南】

Ollama 是一個開源的本地 LLM 執行框架。

針對最新的 Llama 3.2 (3GB) 模型，
其參數大小約為 30 億。

在硬體需求方面，
記憶體（RAM/VRAM）至少需要 4GB 才能勉強運行，

但為了達到流暢的生成速度（每秒 30 字以上），
強烈建議配備 8GB 以上的記憶體。

若是 1GB 模型，
則只需 2GB 記憶體。
"""

# Context（檢索內容）
context = parent_document

# User Question（使用者問題）
question = "Llama 3.2 4GB 跑得動嗎？"

# Prompt（提示模板）
prompt = f"""
請根據以下引文回答問題。

引文：
{context}

問題：
{question}
"""

# Generate Answer（產生回答）
response = ollama.generate(
    model = "llama3.2:latest",
    prompt = prompt
)

# Output Result（輸出結果）
print("輸出結果：")
print(response["response"])