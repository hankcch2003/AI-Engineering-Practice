from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler

print("===== Callback 過濾 + 雙模型輸出控制 =====")

# Token 輸出 + 敏感詞過濾
class TypingCallback(BaseCallbackHandler):
    def __init__(self):
        self.badwords = ["暴力", "生氣", "威脅"]

    def on_llm_new_token(self, token, **kwargs):

        # 預設輸出內容
        display_token = token

        # 敏感詞替換
        for word in self.badwords:
            if word in display_token:
                display_token = display_token.replace(word, "OOO")

        print(display_token, end = "", flush = True)

# LLM 執行流程紀錄
class MySyncCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"\nLLM 啟動，提問：{prompts}")

    def on_llm_end(self, response, **kwargs):
        print()
        print("\nLLM 回應完成\n")

# callbacks
handler1 = TypingCallback()
handler2 = MySyncCallback()

# model 1：stream + token control
llm1 = OllamaLLM(model = "llama3.2:latest")

# model 2：callback log
llm2 = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [handler2]
)

# stream + token filter
llm1.stream("請介紹台灣熱門程式語言", config = {"callbacks": [handler1]})

# normal stream + lifecycle callback
for chunk in llm2.stream("請介紹台灣熱門程式語言"):
    print(chunk, end = "", flush = True)