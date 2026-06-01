import wikipedia
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_ollama import OllamaLLM

print("===== 180_LangChain_Wikipedia_Agent實驗 =====\n")

# Ollama Model（語言模型）
llm = OllamaLLM(model = "llama3.2:latest")

# Wikipedia Search Function（Wikipedia 搜尋函式）
def wiki_search(query):
    try:
        query = query.replace("_", "").strip()
        return wikipedia.summary(query, sentences = 3, auto_suggest = False, redirect = True)

    except Exception:
        return f"查無 Wikipedia 資料：{query}"

# Tools（工具設定）
tools = [
    Tool(
        name = "Wikipedia",
        func = wiki_search,
        description = "useful for getting encyclopedia information"
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
        You are a strict ReAct agent.

        IMPORTANT RULES:
        1. Tool name must be exactly: Wikipedia
        2. Do NOT modify tool name
        3. Action format must be:
        Action: Wikipedia
        Action Input: <query>
        4. Action Input must be English keyword only (no sentences, no punctuation)
        """
    }
)

# Query（查詢問題）
query = "Python 是什麼？"

# Agent Invoke（代理人執行）
response = agent.invoke(query)

# Response（模型輸出）
print("=" * 50)
print("回答結果：")
print(response)