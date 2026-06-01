import warnings
import logging
import ollama

# 忽略警告訊息（避免輸出過多 warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

print("===== 205_MultiQuery_Search實驗 =====\n")

# User Question（使用者問題）
user_question = "生態保育以及環境保護於資訊科學機器學習中有什麼應用？"

# Prompt（多查詢生成提示）
prompt = f"""
角色 (Role)：你是一個嚴謹的搜尋最佳化專家。

上下文 (Context)：
請針對使用者的問題，
生成 10 個不同寫法的繁體中文同義搜尋關鍵字或句子。

指令 (Instruction)：
用來幫助向量資料庫檢索。

問題 / 輸入資料 (Question / Input Data)：
使用者問題：{user_question}

限制 (Constraints)：
每行一個，不要有任何前言或編號。
"""

# Generate Queries（產生多查詢）
response = ollama.generate(
    model = "llama3.2:latest",
    prompt = prompt
)

generated_queries = response["response"].strip().split("\n")

# Print Queries（輸出查詢結果）
print("===== Multi Query Result =====\n")

for i, query in enumerate(generated_queries, start = 1):
    print(f"Query {i}: \n{query}")
    print("=" * 50)