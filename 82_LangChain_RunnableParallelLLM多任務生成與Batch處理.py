from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate

# LLM 模型
llm = OllamaLLM(model = "llama3.2:latest")

# 前處理：關鍵字包裝
preprocess = RunnableLambda(lambda x: f"關鍵字：{x}")

# 生成簡短句子
prompt_short = PromptTemplate.from_template("請生成簡短的句子(20字以內)：{text}")
chain_short = prompt_short | llm | (lambda x: f"[簡短句子] {x}")

# 生成笑話
prompt_joke = PromptTemplate.from_template("請針對 {text} 講一個簡短的笑話(20字以內)")
chain_joke = prompt_joke | llm | (lambda x: f"[笑話] {x}")

# 平行處理
parallel_chain = preprocess | RunnableParallel(
    Sentence = chain_short,
    Joke = chain_joke
)

# 單筆輸入
parallel_result = parallel_chain.invoke("蛋糕")
print(parallel_result)

# 批次輸入
batch_result = parallel_chain.batch([
    "美麗的蓮花",
    "日本",
    "孟寶的感情小教室"
])

# 輸出 batch 結果
for res in batch_result:
    print("Sentence：", res["Sentence"])
    print("Joke：", res["Joke"])
    print()