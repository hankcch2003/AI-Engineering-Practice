import warnings
import logging

# 忽略警告訊息（避免輸出過多 LangChain / Deprecation Warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

print("===== 188_LangChain_DuckDuckGo_Search_Agent實驗 =====\n")

# DuckDuckGo Search Tool（搜尋工具）
search_tool = DuckDuckGoSearchRun()
search_tool.name = "duckduckgo_search"

search_tool.description = """
當需要查詢即時資訊、活動資訊、
新聞事件或未知知識時使用。
"""

# Tools（工具列表）
tools = [search_tool]
tools_dict = {tool.name: tool for tool in tools}

# Ollama Model（語言模型）
llm = ChatOllama(model = "llama3.2:latest", temperature = 0)

# Bind Tools（工具綁定）
llm_with_tools = llm.bind_tools(tools)

# Prompt Template（提示模板）
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位來自台灣的 AI 助理。請使用繁體中文與台灣用語回答。"
    ),
    ("human", "{input}")
])

# Custom Search Agent（自訂搜尋 Agent）
def run_custom_search_agent(user_input: str):
    print(f"使用者問題：{user_input}")

    # Prompt → Messages
    messages = prompt_template.format_messages(input = user_input)

    # First LLM Call（第一次推理）
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    # Tool Calls（工具調用）
    if ai_msg.tool_calls:

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print("=" * 50)
            print(f"[系統偵測] 工具：{tool_name}")
            print()
            print(f"[參數]：{tool_args}")
            print("=" * 50)

            # Execute Tool（執行工具）
            the_tool = tools_dict[tool_name]
            query_val = tool_args.get("query", str(tool_args))

            if isinstance(query_val, dict):
                query_val = list(query_val.values())[0]

            observation = the_tool.invoke(str(query_val))

            print()
            print(f"[搜尋結果]：{observation[:150]}...")
            print("=" * 50)

            messages.append(
                ToolMessage(
                    content = observation,
                    tool_call_id = tool_call["id"]
                )
            )

        print("正在整合搜尋結果並產生最終回答...")

        # Final Answer（最終回答）
        final_msg = llm.invoke(messages)
        return final_msg.content

    else:
        return ai_msg.content

# Query（查詢問題）
final_answer = run_custom_search_agent("2025年6月 台北大巨蛋 職棒 活動")

# Output（輸出結果）
print("=" * 50)
print("回答結果：")
print(final_answer)
print()