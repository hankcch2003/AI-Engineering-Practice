from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import datetime

print("===== Prompt 改寫 + Callback Log 系統 =====")

# custom callback handler（自訂回調處理器）
class LogCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):

        # log start（開始紀錄）
        print("LOG START")

        # time record（時間紀錄）
        print("時間：", datetime.now())

        # type check（型別檢查）
        print(type(prompts))

        # prompt observe（提示詞觀察，不修改模型輸入）
        if isinstance(prompts, list):
            print(f"原始問題：{prompts[0]}")
        else:
            print(f"原始問題：{prompts}")

    def on_llm_end(self, response, **kwargs):
        # log end（結束紀錄）
        print()
        print("回答完成")
        print("時間：", datetime.now())
        print("LOG END\n")

# initialize LLM（初始化模型）
llm = OllamaLLM(model = "llama3.2:latest")

# callback handler（回調處理器）
handler = LogCallback()

# prompt preprocessing（提示詞預處理，改寫提示詞內容）
base_prompt = "請用一句話形容蛋糕"
modified_prompt = "請以川普語氣，自信、誇張、口語化回答：" + base_prompt

# invoke execution（執行調用，帶入改寫後的提示詞和回調處理器）
result = llm.invoke(modified_prompt, config = {"callbacks": [handler]})

# output result（輸出結果）
print("模型回答：", result)