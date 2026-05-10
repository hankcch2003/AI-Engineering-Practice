from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# LLM initialization (LLM 初始化)
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt template (提示詞模板)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位專業的文案寫手，擅長撰寫 SEO 標題。"
        "你只能輸出一行結果，不能有任何解釋或多餘文字。"
        "所有輸出必須使用繁體中文。"
        "嚴禁使用英文或其他語言。"
        "輸出格式必須完全符合：SEO 標題：XXX"
    ),
    (
        "user",
        "請根據以下文章內容生成 SEO 標題：\n文章內容：{article}"
    )
])

# Chain (建立鏈式結構)
chain = prompt | llm

# Streamlit UI (使用者介面)
st.title("SEO標題生成器")

# 使用者輸入區（文章內容）
article = st.text_area("請輸入文章內容：", height = 200)

# 生成按鈕（事件觸發）
if st.button("生成 SEO 標題"):

    # 防呆：避免空輸入
    if not article.strip():
        st.warning("請先輸入內容！")

    else:
        st.subheader("產生結果：")

        # 動態輸出容器（streaming 更新用）
        output_box = st.empty()
        result = ""

        # Streaming output (串流輸出)
        for chunk in chain.stream({"article": article}):
            result += chunk
            output_box.markdown(result)