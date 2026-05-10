from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler

print("===== Callback vs 無 Callback 對照 =====")

# Token 輸出 + 敏感詞過濾
class TypingCallback(BaseCallbackHandler):
    def __init__(self):
        self.badwords = ["暴力", "生氣", "傷害", "傷"]

    def on_llm_new_token(self, token, **kwargs):

        # 預設輸出
        display_token = token

        # 敏感詞替換
        for word in self.badwords:
            if word in display_token:
                display_token = display_token.replace(word, "$$$")

        print(display_token, end = "", flush = True)

# callback handler，會在每次 LLM 輸出新 token 時被呼叫
handler1 = TypingCallback()

# 初始化 OllamaLLM，指定使用的模型
llm1 = OllamaLLM(model = "llama3.2:latest")

# 有 callback（過濾版）
print("===== 有 Callback（過濾輸出） =====")

# 使用 stream 方式輸出，並帶入 callback handler
for chunk in llm1.stream("請告訴我暴力與傷害的相關資訊", config = {"callbacks": [handler1]}):
    pass

# 無 callback（原始輸出）
print("\n")
print("===== 無 Callback（原始輸出） =====")

# 使用 stream 方式輸出，沒有帶入 callback handler
for chunk in llm1.stream("請告訴我暴力與傷害的相關資訊"):
    print(chunk, end = "", flush = True)