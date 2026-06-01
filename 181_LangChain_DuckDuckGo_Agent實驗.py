from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun

print("===== 181_LangChain_DuckDuckGo_Agent實驗 =====\n")

# Ollama Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# DuckDuckGo Search Function（DuckDuckGo 搜尋函式）
search = DuckDuckGoSearchRun()

# Tools（工具設定）
tools = [
    Tool(
        name = "DuckDuckGo",
        func = search.run,
        description = "useful for searching current web information"
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
        You are a strict Taiwanese AI assistant.

        IMPORTANT RULES:
        1. Tool name must be exactly: DuckDuckGo
        2. Do NOT modify tool name
        3. Action format must be:
        Action: DuckDuckGo
        Action Input: <query>
        4. Action Input must be keyword only (no sentences, no punctuation)
        """
    }
)

# Query（查詢問題）
query = "請告訴我2026年6月大巨蛋最近的活動資訊"

# Agent Invoke（代理人執行）
response = agent.invoke(query)

# Response（模型輸出）
print("=" * 50)
print("回答結果：")
print(response)