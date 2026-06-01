import warnings

# 忽略警告訊息（避免輸出過多 LangChain / Deprecation Warning）
warnings.filterwarnings("ignore")

import wikipedia
from langchain_core.tools import tool
from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

print("===== 184_LangChain_Custom_Tool_Calling_Agent實驗 =====\n")

# Wikipedia Tool（維基百科工具）
@tool
def wikipedia_search(query: str) -> str:
    """查詢 Wikipedia 資料"""

    try:
        if isinstance(query, dict):
            query = list(query.values())[0]

        return wikipedia.summary(str(query), sentences = 3)

    except Exception as e:
        return f"維基百科找不到相關資料：{str(e)}"

# Tools（工具列表）
tools = [wikipedia_search]
tools_dict = {t.name: t for t in tools}

# Ollama Model（語言模型）
llm = ChatOllama(model = "llama3.2:latest", temperature = 0)

# Bind Tools（工具綁定）
llm_with_tools = llm.bind_tools(tools)

# Prompt Template（提示模板）
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一位來自台灣的 AI 助理。請務必使用繁體中文與台灣用語回答。"),
    ("human", "{input}")
])

# Agent Function（自訂 Agent 流程）
def run_custom_agent(user_input: str):
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
            print()

            # Execute Tool（執行工具）
            the_tool = tools_dict[tool_name]
            query_val = tool_args.get("query", str(tool_args))
            observation = the_tool.invoke({"query": str(query_val)})

            print(f"[工具結果] {observation}")
            print("=" * 50)

            messages.append(
                ToolMessage(
                    content = observation,
                    tool_call_id = tool_call["id"]
                )
            )

        print("正在產生最終回答...")

        # Final Answer（最終回答）
        final_msg = llm_with_tools.invoke(messages)
        return final_msg.content

    else:
        return ai_msg.content

# Query（查詢問題）
response = run_custom_agent("蘭嶼是甚麼地方？")

# Output（輸出結果）
print("=" * 50)
print("回答結果：")
print(response)