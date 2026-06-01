import warnings
import logging

# 忽略警告訊息（避免輸出過多 LangChain / Deprecation Warning）
warnings.filterwarnings("ignore")

# 隱藏 Pydantic 錯誤訊息
logging.getLogger("pydantic").setLevel(logging.ERROR)

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage

print("===== 187_LangChain_CSV_Tool_Agent實驗 =====\n")

# Read CSV（讀取 CSV）
try:
    df = pd.read_csv("sales.csv")

except FileNotFoundError:
    print("[系統提示] 找不到 sales.csv，已建立測試資料。")

    df = pd.DataFrame({"sales": [100, 200, 300, 400, 500]})

# CSV Analysis Tool（CSV 分析工具）
@tool
def analyze_data(question: str) -> str:
    """
    當需要分析 CSV 資料時使用。
    目前支援：
    1. 平均值
    2. 最大值
    """

    try:
        if isinstance(question, dict):
            question = question.get(
                "question",
                list(question.values())[0]
            )

        question_str = str(question)

        # Mean（平均值）
        if "平均" in question_str:
            mean_val = df["sales"].mean()

            return f"銷售額平均值為：{mean_val}"

        # Max（最大值）
        if "最大" in question_str:
            max_val = df["sales"].max()

            return f"銷售額最大值為：{max_val}"

        return "目前僅支援『平均』與『最大』分析"

    except Exception as e:
        return f"資料分析發生錯誤：{str(e)}"

# Tools（工具列表）
tools = [analyze_data]
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

# Custom Agent（自訂 Agent）
def run_custom_analysis_agent(user_input: str):
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
            question_val = tool_args.get("question", str(tool_args))
            observation = the_tool.invoke({"question": str(question_val)})

            print(f"[工具結果]：{observation}")
            print("=" * 50)

            messages.append(
                ToolMessage(
                    content = observation,
                    tool_call_id = tool_call["id"]
                )
            )

        print("正在產生最終回答...")

        # Final Answer（最終回答）
        final_msg = llm.invoke(messages)
        return final_msg.content

    else:
        return ai_msg.content

# Query（查詢問題）
final_answer = run_custom_analysis_agent("sales 平均是多少？")

# Output（輸出結果）
print("=" * 50)
print("回答結果：")
print(final_answer)