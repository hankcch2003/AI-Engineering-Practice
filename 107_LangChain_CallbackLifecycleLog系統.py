from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import datetime

print("===== Callback Lifecycle Log 系統 =====")

# custom callback handler（自訂回調處理器）
class LogCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):

        # log start（開始紀錄）
        print("LOG START")

        # time record（時間紀錄）
        print("時間：", datetime.now())

        # prompt log（輸入問題紀錄）
        print(f"問題：{prompts}")

    def on_llm_end(self, response, **kwargs):
        # log end（結束紀錄）
        print()
        print("回答完成")
        print("時間：", datetime.now())
        print("LOG END\n")

# initialize LLM（模型初始化）
llm = OllamaLLM(model = "llama3.2:latest")

# initialize callback handler（回調處理器）
handler = LogCallback()

# invoke execution（執行調用，帶入執行後的提示詞和回調處理器）
result = llm.invoke("請用一句話形容蛋糕", config = {"callbacks": [handler]})

# output result（輸出結果）
print("模型回答：", result)