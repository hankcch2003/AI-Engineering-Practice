from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler

print("===== Callback 過濾 + 輸出控制 =====")

# Token 輸出 + 敏感詞替換
class TypingCallback(BaseCallbackHandler):
    def __init__(self):
        self.badwords = ["暴力", "生氣", "威脅"]

    def on_llm_new_token(self, token, **kwargs):

        # 預設輸出內容
        display_token = token

        # 敏感詞替換
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

# model 1：callback 控制輸出
llm1 = OllamaLLM(model = "llama3.2:latest")

# model 2：log callback
llm2 = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [handler2]
)

# invoke + callback control（輸出攔截）
print(llm1.invoke("請告訴我暴力與生氣的相關資訊", config = {"callbacks": [handler1]}))