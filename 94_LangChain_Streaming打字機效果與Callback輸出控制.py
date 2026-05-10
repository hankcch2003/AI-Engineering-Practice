from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler

print("===== Stream Callback 輸出控制 =====")

# Token 輸出 + 敏感詞過濾
class TypingCallback(BaseCallbackHandler):
    def __init__(self):
        self.badwords = ["暴力", "生氣", "威脅"]

    def on_llm_new_token(self, token, **kwargs):

        # default output（預設輸出）
        display_token = token

        # sensitive word filter（敏感詞替換）
        for word in self.badwords:
            if word in display_token:
                display_token = display_token.replace(word, "和平")

        print(display_token, end = "", flush = True)

# LLM lifecycle log
class MySyncCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, query, **kwargs):
        print(f"LLM 啟動，使用的提問：{query}")

    def on_llm_end(self, response, **kwargs):
        print("LLM 回應完成")

# callbacks
handler1 = TypingCallback()
handler2 = MySyncCallback()

# model 1：stream + token control
llm1 = OllamaLLM(model = "llama3.2:latest")

# model 2：lifecycle log
llm2 = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [handler2]
)

# stream + callback filter（輸出攔截）
for chunk in llm1.stream("請告訴我暴力與生氣的相關資訊", config = {"callbacks": [handler1]}):
    pass