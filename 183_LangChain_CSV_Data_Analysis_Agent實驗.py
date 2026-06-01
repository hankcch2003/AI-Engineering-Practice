from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
import pandas as pd

print("===== 183_LangChain_CSV_Data_Analysis_Agent實驗 =====\n")

# Ollama Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Load CSV（讀取資料）
df = pd.read_csv("./sales.csv")

# Data Analysis Function（資料分析函式）
def analyze_data(question):
    if "平均" in question:
        return str(df["sales"].mean())

    if "最大" in question:
        return str(df["sales"].max())

    return "找不到對應分析"

# Tools（工具設定）
tools = [
    Tool(
        name = "Data Analysis",
        func = analyze_data,
        description = "分析 CSV 資料（平均、最大值）"
    )
]

# Agent（代理人）
agent = initialize_agent(
    tools = tools,
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
    handle_parsing_errors = True,
    max_iterations = 3,
    early_stopping_method = "generate",

    agent_kwargs = {
        "prefix": """
        You are a Taiwanese AI assistant.

        Rules:
        1. Final answer must be in Traditional Chinese.
        2. Use Taiwan-style terminology.
        """
    }
)

# Query（查詢問題）
query = "sales 平均是多少？"

# Agent Invoke（代理人執行）
response = agent.invoke(query)

# Response（輸出結果）
print("=" * 50)
print("回答結果：")
print(response)