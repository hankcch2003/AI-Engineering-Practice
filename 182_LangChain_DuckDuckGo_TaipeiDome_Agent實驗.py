from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun

print("===== 182_LangChain_DuckDuckGo_TaipeiDome_Agent實驗 =====\n")

# Ollama Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# DuckDuckGo Search Function（搜尋工具）
search = DuckDuckGoSearchRun()

# Tools（工具設定）
tools = [
    Tool(
        name = "DuckDuckGo",
        func = search.run,
        description = "當你需要查詢台灣即時、最新的時事或活動資訊時使用。輸入(Action Input)必須使用繁體中文關鍵字，例如：2025年6月 台北大巨蛋 活動"
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
        2. Use Taiwanese terminology.
        """
    }
)

# Query（查詢問題）
query = "2025年6月 台北大巨蛋 活動"

# Agent Invoke（代理人執行）
response = agent.invoke(query)

# Response（輸出結果）
print("=" * 50)
print("回答結果：")
print(response)