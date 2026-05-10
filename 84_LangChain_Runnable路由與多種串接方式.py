from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableSequence

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

print("===== 條件路由處理 =====")

# 路由規則
def router(text):
    if "翻譯" in text:
        return "請翻譯：" + text
    else:
        return "請回答：" + text

print("===== pipe 串接 =====")

chain1 = RunnableLambda(router).pipe(llm)

print("輸入：翻譯 Hello")
print("輸出：", chain1.invoke("翻譯 Hello"))
print()

print("輸入：介紹 Python")
print("輸出：", chain1.invoke("介紹 Python"))
print()

print("===== | pipeline 串接 =====")

chain2 = RunnableLambda(router) | llm

print("輸入：翻譯 Hello")
print("輸出：", chain2.invoke("翻譯 Hello"))
print()

print("輸入：介紹 Python")
print("輸出：", chain2.invoke("介紹 Python"))
print()

print("===== RunnableSequence 串接 =====")

chain3 = RunnableSequence(
    first = RunnableLambda(router),
    last = llm
)

print("輸入：翻譯 Hello")
print("輸出：", chain3.invoke("翻譯 Hello"))
print()

print("輸入：介紹 Python")
print("輸出：", chain3.invoke("介紹 Python"))