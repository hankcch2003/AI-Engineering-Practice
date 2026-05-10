from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
import streamlit as st

# LLM initialization (LLM 初始化)
llm = OllamaLLM(model = "llama3.2:latest")

# Prompt template (提示詞模板)
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一位多國語言翻譯助手，能夠將用戶提供的文字翻譯成多種語言。"
        "請根據用戶的要求，翻譯成指定的目標語言。"
        "輸出格式為：\n原文：\n目標語言：\n翻譯："
    ),
    (
        "user",
        "請將以下文字翻譯成 {language}：{original}"
    )
])

# Chain (建立鏈式結構)
chain = prompt | llm

# Callback Handler (Streaming)
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# Streamlit UI (使用者介面)
st.title("中文翻譯工具")

# 使用者輸入區（原文）
article = st.text_area("請輸入需要翻譯的文字", height = 200)

# 目標語言選擇器
target_language = st.selectbox("目標語言", ["英文", "日文", "法文", "韓文", "阿拉伯文"])

# 翻譯觸發按鈕（事件入口）
if st.button("進行翻譯"):

    # 防呆：避免空輸入
    if not article.strip():
        st.warning("請先輸入內容！")

    else:
        # 結果區標題
        st.subheader("翻譯結果：")

        # 動態輸出容器（streaming 更新用）
        output_box = st.empty()

        # 初始化 streaming callback handler
        handler = StreamHandler(output_box)

        # Chain execution (執行模型)
        chain.invoke(
            {"original": article, "language": target_language},
            {"callbacks": [handler]}
        )