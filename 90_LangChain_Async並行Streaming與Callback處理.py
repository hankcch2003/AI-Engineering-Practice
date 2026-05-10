import asyncio
from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import AsyncCallbackHandler

print("===== Async LLM 並行處理 =====")

# 非同步 Callback
class MyAsyncCallback(AsyncCallbackHandler):
    async def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"（非同步）LLM 開始處理：{prompts}")

    async def on_llm_end(self, response, **kwargs):
        print()
        print("\n（非同步）LLM 處理完成\n")

# Callback handler
callback_handler = MyAsyncCallback()

# LLM 模型
llm = OllamaLLM(
    model = "llama3.2:latest",
    callbacks = [callback_handler]
)

# 單一 task
async def run_llm_task(prompt):
    async for chunk in llm.astream(prompt):
        print(chunk, end = "", flush = True)

# 主程式
async def main():
    tasks = [
        run_llm_task("台灣五月有什麼地方可以旅行"),
        run_llm_task("機器學習和深度學習有什麼區別？"),
        run_llm_task("微積分在生活上有什麼應用？")
    ]
    await asyncio.gather(*tasks)

# 執行 async event loop
asyncio.run(main())