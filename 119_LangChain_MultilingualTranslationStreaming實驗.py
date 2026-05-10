from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

print("===== Multilingual Translation Streaming 實驗 =====")

# LLM initialization (LLM 初始化)
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt template (提示詞模板)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位多國語言翻譯助手，能夠將用戶提供的文字翻譯成多種語言。"
     "請根據用戶的要求，翻譯成指定的目標語言。"
     "輸出必須嚴格符合以下格式，不能多任何字，且每一行格式需對齊：\n"
     "原文：...\n"
     "目標語言：...\n"
     "結果：..."),
    ("user",
     "請將以下文字翻譯成 {language}：{original}")
])

# Input (使用者輸入)
user_text = input("請輸入需要翻譯的文字：")
target_language = input("請輸入目標語言 (英文 / 日文 / 韓文)：")

print()
print("正在進行翻譯...\n")

# Chain creation (建立鏈式結構)
chain = prompt | llm

# Model invocation (呼叫模型)
chain.invoke(
    {"original": user_text, "language": target_language},
    {"callbacks": [StreamingStdOutCallbackHandler()]}
)