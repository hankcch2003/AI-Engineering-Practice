from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

# 計算字數
def count_words(text: str) -> str:
    length = len(text)
    return f"{text}\n\n[字數統計：{length} 字]"

# Runnable 後處理：字數統計
word_counter = RunnableLambda(count_words)

# Prompt + LLM + 後處理流程
chain = (
    ChatPromptTemplate.from_template("請簡短介紹：{topic}")
    | llm
    | word_counter
)

# 執行 chain
result = chain.invoke({"topic": "玉山"})
print(result)