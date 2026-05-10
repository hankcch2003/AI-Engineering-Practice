import time
from langchain_ollama import OllamaLLM
from langchain_core.callbacks.base import BaseCallbackHandler

print("===== Advanced Callback 控制系統 =====")

class AdvancedControlCallback(BaseCallbackHandler):
    def __init__(self):

        # 敏感詞列表
        self.bad_words = ["暴力", "生氣", "傷害", "報復", "憤怒", "焦慮"]

        # buffer 控制
        self.buffer = ""
        self.look_back = 4

        # 統計
        self.token_count = 0
        self.start_time = time.time()

        # 設定輸出紀錄檔案
        self.log_file = open("101_LLM輸出紀錄_log.txt", "a", encoding = "utf-8")

    def on_llm_new_token(self, token: str, **kwargs):
        # token 計數（更準）
        self.token_count += 1

        # debug 原始 token
        print(f"[{token}]", end = "", flush = True)

        # buffer 累積
        self.buffer += token

        # 敏感詞過濾
        for word in self.bad_words:
            if word in self.buffer:
                self.buffer = self.buffer.replace(word, "***")

        # buffer flush 控制
        if len(self.buffer) > self.look_back:
            ready_content = self.buffer[:-self.look_back]
            self.buffer = self.buffer[-self.look_back:]
            self._process_output(ready_content)

    def _process_output(self, content: str):
        if not content:
            return

        # 輸出排版
        print(content, end = "", flush = True)

        # 紀錄到檔案
        self.log_file.write(content)
        self.log_file.flush()

    def on_llm_end(self, response, **kwargs):
        # flush 剩餘 buffer
        if self.buffer:
            self._process_output(self.buffer)
            self.buffer = ""

        # 統計
        total_time = time.time() - self.start_time
        speed = self.token_count / total_time if total_time > 0 else 0

        print("\n")
        print("===== 統計報告 =====")
        print(f"Token 數：{self.token_count}")
        print(f"耗時：{total_time:.2f} 秒")
        print(f"速度：{speed:.2f} tokens/sec")

        self.log_file.close()

# model
llm = OllamaLLM(model = "llama3.2:latest")

# handler
handler = AdvancedControlCallback()

print("開始精確控制輸出")
print()

# 直接使用 stream 模式，並傳入 callback handler
for _ in llm.stream("請告訴我關於生氣與暴力的處理方式", config = {"callbacks": [handler]}):
    pass