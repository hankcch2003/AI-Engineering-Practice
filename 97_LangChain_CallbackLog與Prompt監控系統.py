from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler
from datetime import datetime

print("===== Callback Log + Prompt 監控系統 =====")

# Log callback（流程紀錄 + prompt 監控）
class LogCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):

        # log start（開始紀錄）
        print("LOG START")

        # time record（時間紀錄）
        print("時間：", datetime.now())

        # prompts safety handling（避免 list 錯誤）
        if isinstance(prompts, list):
            prompt_text = prompts[0] if prompts else ""
        else:
            prompt_text = prompts

        # prompt 改寫觀察
        prompt_text = "請以文言文方式回應：" + prompt_text
        print("問題：", prompt_text)
        print()

    def on_llm_end(self, response, **kwargs):
        # log end（結束紀錄）
        print("回答完成")
        print("LOG END\n")

# LLM model
llm = OllamaLLM(model = "llama3.2:latest")

# callback handler
handler = LogCallback()

# invoke with callback（開啟監控）
result = llm.invoke("請用一句話形容蛋糕", config = {"callbacks": [handler]})
print("模型回答1：", result)

# invoke without callback（一般模式）
result = llm.invoke("請用一句話形容蛋糕")
print("模型回答2：", result)