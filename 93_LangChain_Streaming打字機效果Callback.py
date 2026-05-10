from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler

print("===== Token Streaming（打字機效果） =====")

# Callback：逐 token 輸出
class TypingCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):

        # 每個 token 即時輸出
        print(token, end = "", flush = True)

# 開啟 streaming 模式
llm = OllamaLLM(
    model = "llama3.2:latest",
    streaming = True
)

# Callback handler
handler = TypingCallback()

# 執行 streaming（會逐 token 觸發 callback）
llm.invoke("請介紹台灣熱門程式語言", config = {"callbacks": [handler]})