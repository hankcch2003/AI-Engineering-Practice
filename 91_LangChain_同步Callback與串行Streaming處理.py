from langchain_ollama import OllamaLLM
from langchain_core.callbacks import BaseCallbackHandler

print("===== 同步 Callback + 串行 Streaming =====")

# 同步 Callback
class MySyncCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"（同步）LLM 開始處理：{prompts}")

    def on_llm_end(self, response, **kwargs):
        print()
        print("\n（同步）LLM 處理完成")

# Callback handler
callback_handler = MySyncCallback()

# LLM 模型
llm = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [callback_handler]
)

# 單一任務
def run_llm_task(prompt):
    print(f"\n使用者問題：{prompt}\n")

    for chunk in llm.stream(prompt):
        print(chunk, end = "", flush = True)

# 主流程
def main():
    prompts = [
        "台灣五月有什麼地方可以旅行",
        "機器學習和深度學習有什麼區別？",
        "微積分生活上有什麼運用"
    ]

    for prompt in prompts:
        run_llm_task(prompt)

# 執行
if __name__ == "__main__":
    main()