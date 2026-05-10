from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import datetime

print("===== Prompt + Callback 整合系統 =====")

# custom callback handler（自訂回調處理器）
class LogCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):

        # log start（開始紀錄）
        print("LOG START")

        # time record（時間紀錄）
        print("時間：", datetime.now())

        # prompt observe（提示詞觀察）
        print(f"問題：{prompts}")

    def on_llm_end(self, response, **kwargs):
        # log end（結束紀錄）
        print()
        print("回答完成")
        print("時間：", datetime.now())
        print("LOG END\n")

# initialize LLM（初始化模型）
llm = OllamaLLM(model = "llama3.2:latest")

# initialize callback handler（回調處理器）
handler = LogCallback()

# invoke execution（執行調用，帶入改寫後的提示詞和回調處理器）
result = llm.invoke("請以川普語氣，自信、誇張、口語化回答：請用一句話形容蛋糕", config = {"callbacks": [handler]})

# output result（輸出結果）
print("模型回答：", result)