from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

print("===== SEO Streaming 標題生成實驗 =====")

# LLM initialization (LLM 初始化)
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt template (提示詞模板)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一位專業的文案寫手，擅長撰寫 SEO 標題。"
     "你只能輸出一行結果，不能有任何解釋或多餘文字。"
     "輸出格式必須完全符合：SEO 標題：XXX"),
    ("user",
     "請根據以下文章內容生成 SEO 標題：\n文章內容：{article}")
])

# Input (使用者輸入)
user_article = input("請輸入文章內容：")

# Chain (建立鏈式結構)
chain = prompt | llm

print()
print("正在生成 SEO 標題...\n")
print("生成的 SEO 標題：", end = "")

# Streaming output (串流輸出)
for chunk in chain.stream({"article": user_article}):
    print(chunk, end = "", flush = True)