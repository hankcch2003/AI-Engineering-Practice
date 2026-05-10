from langchain_ollama import OllamaLLM
from langchain_core.callbacks import BaseCallbackHandler

# 自訂 Callback
class MySyncCallback(BaseCallbackHandler):

    # LLM 開始
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 啟動，提問：{prompts}")

    # LLM 結束
    def on_llm_end(self, response, **kwargs):
        print()
        print("\nLLM 回應完成")

print("===== LLM Callback + Streaming 監聽 =====")

# Callback handler
callback_handler = MySyncCallback()

# LLM 模型
llm = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [callback_handler]
)

# Streaming 輸出
for chunk in llm.stream("台灣的熱門程式語言有哪些？"):
    print(chunk, end = "", flush = True)